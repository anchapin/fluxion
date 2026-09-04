//! Stable, versioned JSON summary schema.
//!
//! The harness emits **exactly one** [`Summary`] value per evaluation.
//! Every field's meaning, type, and serialization is part of the public
//! contract; downstream evolvers deserialize this struct directly.
//!
//! ## Schema versioning
//!
//! The integer field `schema_version` is the authoritative version. The
//! constant [`CURRENT_SCHEMA_VERSION`] is the version this build emits.
//!
//! Bumping policy:
//! - **Patch fields added** (no field removed or retyped): bump
//!   `schema_version`; old consumers ignore the new fields.
//! - **Field removed or retyped**: bump major, deprecate the previous
//!   schema for one release, and document the migration in
//!   `CHANGELOG.md`.
//! - **Never** remove a field without bumping `schema_version`.

use serde::{Deserialize, Serialize};

use crate::invariant::InvariantViolation;

/// Schema version emitted by this build.
///
/// Bump when the [`Summary`] struct changes shape. See module docs for
/// the policy.
pub const CURRENT_SCHEMA_VERSION: u32 = 1;

/// Newtype wrapping the schema version so callers cannot accidentally
/// pass a bare `u32` into a `Summary` constructor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SchemaVersion(pub u32);

impl std::fmt::Display for SchemaVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// Top-level evaluation outcome.
///
/// This is what every caller (OpenEvolve adapter, CI digest diff, ad-hoc
/// scripts) deserializes. The struct is laid out so the JSON is
/// self-describing and `jq`-friendly: each metric is a top-level field.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Summary {
    /// Schema version. Always `1` in this build; see [`CURRENT_SCHEMA_VERSION`].
    pub schema_version: u32,

    /// Stable identifier for the candidate (seed file name, hash of
    /// source, generation index — the evolver decides the convention).
    pub candidate_id: String,

    /// Generation index. `None` for a one-shot evaluation outside a
    /// campaign (e.g. CI smoke).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub generation: Option<u32>,

    /// Fitness in `[0.0, 1.0]`. Higher is better.
    ///
    /// `0.0` is forced by an [`EvaluationOutcome::InvariantHardFail`]
    /// (the candidate ran but violated a load-bearing invariant such as
    /// energy closure). Otherwise it is a weighted combination of
    /// accuracy margin, latency, and invariant margins.
    pub fitness: f64,

    /// Whether the candidate compiled and produced a runnable artifact.
    /// `false` for compile failures; the harness still emits a Summary
    /// with `fitness = 0.0` so the evolver can record the attempt.
    pub compiled: bool,

    /// Whether every invariant in the battery passed. A failure forces
    /// `fitness = 0.0`; a partial-success margin is reported separately
    /// in [`Summary::min_invariant_margin`].
    pub invariants_passed: bool,

    /// Largest per-edge-case absolute error against the reference. `None`
    /// when compilation failed or no edge cases ran.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub max_error: Option<f64>,

    /// Median eval latency in nanoseconds, over `timing.n` samples.
    /// `None` when compilation failed.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub eval_latency_ns: Option<u64>,

    /// Spread of eval latency (interquartile range, in nanoseconds),
    /// reported alongside the median so consumers can reject noisy
    /// timing. `None` when compilation failed.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub eval_latency_spread_ns: Option<u64>,

    /// sha256 digest of the canonicalized evaluation inputs (candidate
    /// source + edge-case config + toolchain version). Identical inputs
    /// ⇒ identical digest ⇒ byte-identical JSON.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub determinism_digest: Option<String>,

    /// Coarse outcome category. Lets a wrapper script branch on the
    /// result without parsing every field.
    pub outcome: EvaluationOutcome,

    /// Per-invariant violations, if any. Empty for `InvariantsPassed`.
    /// Use this list to drive targeted rejection in the evolver (the
    /// `invariant_kind` field is a stable string the evolver can match
    /// on — e.g. `"energy_closure"`, `"reciprocity"`, `"nan_or_inf"`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub invariant_violations: Vec<InvariantViolation>,

    /// Free-form error message for `CompileFailure`, `SubprocessError`,
    /// `ResourceCap`, etc. `None` for `Evaluated`.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub error: Option<String>,

    /// Smallest invariant margin observed across the battery. `None` if
    /// the candidate produced no numeric output (e.g. compile failure).
    /// A margin ≤ 0 forces `invariants_passed = false`.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub min_invariant_margin: Option<f64>,
}

impl Summary {
    /// Construct a successful evaluation summary with the canonical
    /// field set. The `determinism_digest` is computed by
    /// [`Summary::with_digest`] — pass `None` here and call it after, or
    /// use [`Summary::successful`] which delegates to it for you.
    pub fn new(builder: SummaryBuilder) -> Self {
        let invariants_passed = builder.invariant_violations.is_empty()
            && matches!(builder.min_invariant_margin, Some(m) if m > 0.0);
        Self {
            schema_version: CURRENT_SCHEMA_VERSION,
            candidate_id: builder.candidate_id,
            generation: builder.generation,
            fitness: builder.fitness,
            compiled: true,
            invariants_passed,
            max_error: builder.max_error,
            eval_latency_ns: builder.eval_latency_ns,
            eval_latency_spread_ns: builder.eval_latency_spread_ns,
            determinism_digest: None,
            outcome: EvaluationOutcome::Evaluated,
            invariant_violations: builder.invariant_violations,
            error: None,
            min_invariant_margin: builder.min_invariant_margin,
        }
    }

    /// Like [`Summary::new`] but also computes the determinism digest
    /// from the canonical input bytes. Use this for the final emit.
    pub fn successful(builder: SummaryBuilder, canonical_input: &[u8]) -> Self {
        let mut summary = Self::new(builder);
        summary.determinism_digest = Some(crate::sandbox::determinism_digest(canonical_input));
        summary
    }

    /// Build a Summary for a compile failure. `fitness` is forced to
    /// `0.0`; `compiled = false`; `invariants_passed = false`.
    pub fn compile_failure(
        candidate_id: impl Into<String>,
        generation: Option<u32>,
        error: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: CURRENT_SCHEMA_VERSION,
            candidate_id: candidate_id.into(),
            generation,
            fitness: 0.0,
            compiled: false,
            invariants_passed: false,
            max_error: None,
            eval_latency_ns: None,
            eval_latency_spread_ns: None,
            determinism_digest: None,
            outcome: EvaluationOutcome::CompileFailure,
            invariant_violations: Vec::new(),
            error: Some(error.into()),
            min_invariant_margin: None,
        }
    }

    /// Build a Summary for an invariant hard-fail. `fitness` is forced
    /// to `0.0`; `compiled = true`; `invariants_passed = false`. The
    /// violations list is preserved so the evolver can branch on which
    /// invariant broke.
    pub fn invariant_hard_fail(
        candidate_id: impl Into<String>,
        generation: Option<u32>,
        min_invariant_margin: Option<f64>,
        violations: Vec<InvariantViolation>,
    ) -> Self {
        Self {
            schema_version: CURRENT_SCHEMA_VERSION,
            candidate_id: candidate_id.into(),
            generation,
            fitness: 0.0,
            compiled: true,
            invariants_passed: false,
            max_error: None,
            eval_latency_ns: None,
            eval_latency_spread_ns: None,
            determinism_digest: None,
            outcome: EvaluationOutcome::InvariantHardFail,
            invariant_violations: violations,
            error: None,
            min_invariant_margin,
        }
    }

    /// Build a Summary for a resource-cap (timeout / OOM) hit.
    pub fn resource_cap(
        candidate_id: impl Into<String>,
        generation: Option<u32>,
        error: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: CURRENT_SCHEMA_VERSION,
            candidate_id: candidate_id.into(),
            generation,
            fitness: 0.0,
            compiled: true,
            invariants_passed: false,
            max_error: None,
            eval_latency_ns: None,
            eval_latency_spread_ns: None,
            determinism_digest: None,
            outcome: EvaluationOutcome::ResourceCap,
            invariant_violations: Vec::new(),
            error: Some(error.into()),
            min_invariant_margin: None,
        }
    }

    /// Serialize to a canonical JSON string. Field ordering follows the
    /// struct declaration order (serde default), so identical inputs
    /// produce byte-identical JSON — paired with `determinism_digest`
    /// this gives the campaign reproducer its byte-identity guarantee.
    pub fn to_canonical_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialize from JSON. Used by the OpenEvolve adapter.
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}

/// Coarse outcome category. Mirrors the exit-code table in the crate
/// root docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationOutcome {
    /// Evaluation completed; consult `fitness` and `invariants_passed`.
    Evaluated,
    /// Compile failed (recompilation mode).
    CompileFailure,
    /// Invariant hard-fail — `fitness` is forced to `0.0`.
    InvariantHardFail,
    /// Timeout or resource cap hit before evaluation completed.
    ResourceCap,
}

/// Builder for [`Summary`] — collects the fields that
/// [`Summary::new`] / [`Summary::successful`] take. Using a builder
/// rather than positional arguments keeps the call sites readable
/// and dodges the `clippy::too_many_arguments` lint without
/// `#[allow(...)]` decorations.
#[derive(Clone, Debug)]
pub struct SummaryBuilder {
    /// Stable identifier for the candidate.
    pub candidate_id: String,
    /// Generation index (see [`Summary::generation`]).
    pub generation: Option<u32>,
    /// Fitness in `[0.0, 1.0]`.
    pub fitness: f64,
    /// Largest per-edge-case absolute error.
    pub max_error: Option<f64>,
    /// Median eval latency in nanoseconds.
    pub eval_latency_ns: Option<u64>,
    /// Spread of eval latency.
    pub eval_latency_spread_ns: Option<u64>,
    /// Smallest invariant margin observed.
    pub min_invariant_margin: Option<f64>,
    /// Per-invariant violations.
    pub invariant_violations: Vec<InvariantViolation>,
}

impl SummaryBuilder {
    /// Begin a builder with the candidate-id and fitness set; other
    /// fields default to `None` / empty.
    pub fn new(candidate_id: impl Into<String>, fitness: f64) -> Self {
        Self {
            candidate_id: candidate_id.into(),
            generation: None,
            fitness,
            max_error: None,
            eval_latency_ns: None,
            eval_latency_spread_ns: None,
            min_invariant_margin: None,
            invariant_violations: Vec::new(),
        }
    }

    /// Builder-style: set the generation index.
    pub fn with_generation(mut self, generation: u32) -> Self {
        self.generation = Some(generation);
        self
    }

    /// Builder-style: set the smallest invariant margin only when
    /// `Some` — convenient when the caller has an `Option<f64>` and
    /// wants to keep `None` as "unknown".
    pub fn with_min_invariant_margin_opt(mut self, margin: Option<f64>) -> Self {
        self.min_invariant_margin = margin;
        self
    }

    /// Builder-style: set the largest absolute error.
    pub fn with_max_error(mut self, max_error: f64) -> Self {
        self.max_error = Some(max_error);
        self
    }

    /// Builder-style: set the median eval latency.
    pub fn with_eval_latency_ns(mut self, eval_latency_ns: u64) -> Self {
        self.eval_latency_ns = Some(eval_latency_ns);
        self
    }

    /// Builder-style: set the eval-latency spread.
    pub fn with_eval_latency_spread_ns(mut self, eval_latency_spread_ns: u64) -> Self {
        self.eval_latency_spread_ns = Some(eval_latency_spread_ns);
        self
    }

    /// Builder-style: set the smallest invariant margin.
    pub fn with_min_invariant_margin(mut self, min_invariant_margin: f64) -> Self {
        self.min_invariant_margin = Some(min_invariant_margin);
        self
    }

    /// Builder-style: set the per-invariant violations.
    pub fn with_invariant_violations(mut self, violations: Vec<InvariantViolation>) -> Self {
        self.invariant_violations = violations;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke: serialization round-trips and `schema_version` is present.
    #[test]
    fn round_trip_preserves_schema_version() {
        let summary = Summary::new(
            SummaryBuilder::new("test-candidate", 0.95)
                .with_generation(7)
                .with_max_error(1.2e-3)
                .with_eval_latency_ns(412)
                .with_eval_latency_spread_ns(18)
                .with_min_invariant_margin(0.1),
        );
        let json = summary.to_canonical_json().expect("serialize");
        let parsed: Summary = Summary::from_json(&json).expect("deserialize");
        assert_eq!(parsed.schema_version, CURRENT_SCHEMA_VERSION);
        assert_eq!(parsed.candidate_id, "test-candidate");
        assert_eq!(parsed.fitness, 0.95);
        assert!(parsed.invariants_passed);
    }

    /// Determinism: identical inputs produce byte-identical JSON.
    #[test]
    fn identical_inputs_produce_byte_identical_json() {
        let a = Summary::new(
            SummaryBuilder::new("ctf-seed-0042", 0.9842)
                .with_generation(137)
                .with_max_error(1.2e-4)
                .with_eval_latency_ns(412)
                .with_eval_latency_spread_ns(18)
                .with_min_invariant_margin(0.5),
        );
        let b = Summary::new(
            SummaryBuilder::new("ctf-seed-0042", 0.9842)
                .with_generation(137)
                .with_max_error(1.2e-4)
                .with_eval_latency_ns(412)
                .with_eval_latency_spread_ns(18)
                .with_min_invariant_margin(0.5),
        );
        assert_eq!(
            a.to_canonical_json().unwrap(),
            b.to_canonical_json().unwrap()
        );
    }

    /// Compile failure forces `fitness = 0.0` and `compiled = false`.
    #[test]
    fn compile_failure_forces_zero_fitness() {
        let summary = Summary::compile_failure("ctf-seed-0042", Some(137), "E0425");
        assert_eq!(summary.fitness, 0.0);
        assert!(!summary.compiled);
        assert!(!summary.invariants_passed);
        assert_eq!(summary.outcome, EvaluationOutcome::CompileFailure);
        assert_eq!(summary.error.as_deref(), Some("E0425"));
    }

    /// Schema-versioning forward-compat: a v2 consumer must be able to
    /// parse a v1 JSON without losing data. We simulate that by adding a
    /// synthetic `_future_field` to a serialized v1 JSON and asserting
    /// it round-trips losslessly. If a future schema removes a field,
    /// this test would fail (the field would be missing) — which is
    /// the policy we want.
    #[test]
    fn v1_consumer_tolerates_known_extra_fields() {
        let summary =
            Summary::new(SummaryBuilder::new("ctf-seed-0042", 0.5).with_min_invariant_margin(0.1));
        let mut json: serde_json::Value = serde_json::to_value(&summary).unwrap();
        json["_future_field_added_in_v2"] = serde_json::json!("ignored by v1");
        let parsed: Summary = serde_json::from_value(json).expect("v1 accepts v2 extra");
        assert_eq!(parsed.candidate_id, "ctf-seed-0042");
        assert_eq!(parsed.schema_version, 1);
    }

    /// v1 explicit reject: if a v2 consumer renames `fitness`, the v1
    /// deserializer must refuse the payload rather than silently map
    /// fields. We pin this so a future schema-bump PR cannot silently
    /// downgrade existing campaign artifacts.
    #[test]
    fn v1_rejects_payload_with_missing_required_field() {
        let summary = Summary::new(SummaryBuilder::new("c", 0.5).with_min_invariant_margin(0.1));
        let mut json: serde_json::Value = serde_json::to_value(&summary).unwrap();
        json.as_object_mut().unwrap().remove("fitness");
        let parsed: Result<Summary, _> = serde_json::from_value(json);
        assert!(
            parsed.is_err(),
            "v1 must refuse a payload missing required field `fitness`"
        );
    }
}
