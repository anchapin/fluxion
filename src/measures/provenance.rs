// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT.

//! Provenance tracking for applied Deltas and Python Measures (Issue #1816).
//!
//! When a [`crate::measures::model::FluxionModel`] is mutated — either by a
//! JSON Patch (RFC 6902, Issue #1811) or by a Python Measure executed via the
//! AOT runner (Issue #1814) — the resulting output dataset must be able to
//! answer the question *"why did this model change?"*. This module provides
//! the data structures that make the mutation chain reconstructable.
//!
//! # The provenance contract
//!
//! Every successful mutation appends an [`AppliedDelta`] entry to
//! [`crate::measures::model::FluxionModel::applied_deltas`], in application
//! order. Downstream consumers (ML feature pipelines, debugging tools,
//! reproducibility audits) read the chain back from the serialized model /
//! `results.json` to reconstruct exactly which permutations produced a given
//! energy-use number.
//!
//! # Determinism
//!
//! The `timestamp` field is a **monotonic logical clock** (a zero-padded
//! sequence index assigned from `applied_deltas.len()` at append time), not a
//! wall-clock instant. This is deliberate: the Fluxion determinism gate
//! (Issue #1351) requires that two byte-identical inputs produce byte-identical
//! serialized output, and `SystemTime::now()` would violate that contract.
//! Callers that need a wall-clock instant (e.g. the Python AOT runner) are free
//! to overwrite `timestamp` with an ISO-8601 string at the application layer,
//! where the determinism gate does not apply.
//!
//! # The `AppliedDelta` schema
//!
//! ```json
//! {
//!   "source": "json_patch",
//!   "name": "json_patch:a1b2c3d4e5f60718",
//!   "timestamp": "0000000000000000",
//!   "digest": "a1b2c3d4e5f60718..."
//! }
//! ```
//!
//! - `source` — `"json_patch"` or `"python_measure"` ([`DeltaSource`]).
//! - `name`   — identifier of the patch file / measure class.
//! - `timestamp` — logical sequence (pure-Rust path) or ISO-8601 (Python path).
//! - `digest`  — optional SHA-256 of the patch payload (hex). `null` when the
//!   mutation has no deterministic payload to hash (e.g. a Python measure that
//!   mutates via the snapshot API).

use json_patch::Patch;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The subsystem that produced a mutation.
///
/// Serialised as `snake_case` so downstream JSON consumers can pattern-match on
/// a stable string regardless of Rust rename refactors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeltaSource {
    /// A declarative JSON Patch (RFC 6902) applied via
    /// [`crate::measures::json_patch::apply_delta`] (Issue #1811).
    JsonPatch,
    /// A Python Measure executed by the AOT runner (Issue #1814).
    PythonMeasure,
}

impl DeltaSource {
    /// The stable string used in the `source` JSON field.
    pub fn as_str(self) -> &'static str {
        match self {
            DeltaSource::JsonPatch => "json_patch",
            DeltaSource::PythonMeasure => "python_measure",
        }
    }
}

/// A single provenance entry — one mutation applied to a
/// [`crate::measures::model::FluxionModel`].
///
/// Entries are appended in application order, so `applied_deltas[i]` describes
/// the `i`-th mutation. See the module-level docs for the determinism rationale
/// behind the `timestamp` field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppliedDelta {
    /// What kind of mutation produced this entry.
    pub source: DeltaSource,
    /// Identifier of the patch file or Python measure class
    /// (e.g. `"json_patch:a1b2…"` or `"AddSouthOverhang"`).
    pub name: String,
    /// Logical sequence index (zero-padded, pure-Rust path) or an ISO-8601
    /// wall-clock string (Python AOT-runner path).
    pub timestamp: String,
    /// Optional SHA-256 (hex) of the patch payload. `None` when there is no
    /// deterministic payload to hash.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
}

impl AppliedDelta {
    /// Construct an entry for a JSON Patch mutation.
    ///
    /// `name` is the caller-supplied identifier (e.g. a patch file path);
    /// `digest` is the SHA-256 of the patch payload (see [`digest_of_patch`]).
    /// `seq` is the logical sequence index — normally
    /// `model.applied_deltas.len()` at append time.
    pub fn new_json_patch(name: impl Into<String>, digest: Option<String>, seq: usize) -> Self {
        Self {
            source: DeltaSource::JsonPatch,
            name: name.into(),
            timestamp: logical_timestamp(seq),
            digest,
        }
    }

    /// Construct an entry for a Python Measure mutation.
    ///
    /// Python measures mutate the PyO3 model via the snapshot API, so there is
    /// no deterministic patch payload to hash — `digest` is accepted but is
    /// usually `None` for this source.
    pub fn new_python_measure(name: impl Into<String>, digest: Option<String>, seq: usize) -> Self {
        Self {
            source: DeltaSource::PythonMeasure,
            name: name.into(),
            timestamp: logical_timestamp(seq),
            digest,
        }
    }

    /// Return the short (16-char) prefix of the digest, or `"unnamed"` if the
    /// digest is absent. Useful for building human-readable default names.
    pub fn short_digest_or(&self, fallback: &str) -> String {
        match &self.digest {
            Some(d) if d.len() >= 16 => d[..16].to_string(),
            Some(d) => d.clone(),
            None => fallback.to_string(),
        }
    }
}

/// Render a logical timestamp for sequence index `seq`.
///
/// A zero-padded 16-digit decimal string. This sorts lexicographically in the
/// same order as the numeric sequence, so `applied_deltas` chains compare
/// correctly when serialized.
pub fn logical_timestamp(seq: usize) -> String {
    format!("{:016}", seq)
}

/// Compute the SHA-256 digest (64-char lowercase hex) of a JSON Patch payload.
///
/// The patch is serialised via `serde_json`, whose field ordering is fixed by
/// the `json-patch` crate's `#[derive(Serialize)]`, so two structurally-equal
/// patches always produce the same digest. Returns `None` only if serialisation
/// fails (which, for a `Patch`, is effectively never).
pub fn digest_of_patch(patch: &Patch) -> Option<String> {
    let bytes = serde_json::to_vec(patch).ok()?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let hash = hasher.finalize();
    Some(hash.iter().map(|b| format!("{:02x}", b)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn delta_source_serializes_snake_case() {
        assert_eq!(
            serde_json::to_string(&DeltaSource::JsonPatch).unwrap(),
            r#""json_patch""#
        );
        assert_eq!(
            serde_json::to_string(&DeltaSource::PythonMeasure).unwrap(),
            r#""python_measure""#
        );
    }

    #[test]
    fn delta_source_round_trips() {
        for src in [DeltaSource::JsonPatch, DeltaSource::PythonMeasure] {
            let s = serde_json::to_string(&src).unwrap();
            let back: DeltaSource = serde_json::from_str(&s).unwrap();
            assert_eq!(src, back);
        }
    }

    #[test]
    fn applied_delta_serializes_with_optional_digest() {
        let with_digest = AppliedDelta::new_json_patch("p", Some("abc".to_string()), 0);
        let v: serde_json::Value = serde_json::to_value(&with_digest).unwrap();
        assert_eq!(v["digest"], json!("abc"));
        assert_eq!(v["source"], json!("json_patch"));

        let no_digest = AppliedDelta::new_python_measure("m", None, 1);
        let v: serde_json::Value = serde_json::to_value(&no_digest).unwrap();
        // digest is skipped when None.
        assert!(v.get("digest").is_none() || v["digest"].is_null());
        assert_eq!(v["source"], json!("python_measure"));
    }

    #[test]
    fn logical_timestamp_is_zero_padded_and_sorts() {
        assert_eq!(logical_timestamp(0), "0000000000000000");
        assert_eq!(logical_timestamp(7), "0000000000000007");
        // Lexicographic order matches numeric order for equal-width strings.
        assert!(logical_timestamp(3) < logical_timestamp(12));
    }

    #[test]
    fn digest_of_patch_is_deterministic() {
        let p1: Patch = serde_json::from_value(json!([
            { "op": "replace", "path": "/zones/zone_1/volume", "value": 200.0 }
        ]))
        .unwrap();
        let p2: Patch = serde_json::from_value(json!([
            { "op": "replace", "path": "/zones/zone_1/volume", "value": 200.0 }
        ]))
        .unwrap();
        let d1 = digest_of_patch(&p1).unwrap();
        let d2 = digest_of_patch(&p2).unwrap();
        assert_eq!(d1, d2, "identical patches must hash identically");
        assert_eq!(d1.len(), 64, "SHA-256 hex is 64 chars");
    }

    #[test]
    fn digest_differs_for_different_patches() {
        let p1: Patch = serde_json::from_value(json!([
            { "op": "replace", "path": "/zones/zone_1/volume", "value": 200.0 }
        ]))
        .unwrap();
        let p2: Patch = serde_json::from_value(json!([
            { "op": "replace", "path": "/zones/zone_1/volume", "value": 300.0 }
        ]))
        .unwrap();
        assert_ne!(digest_of_patch(&p1).unwrap(), digest_of_patch(&p2).unwrap());
    }

    #[test]
    fn short_digest_or_fallbacks() {
        let d = AppliedDelta::new_json_patch("p", Some("abcdef0123456789extra".to_string()), 0);
        assert_eq!(d.short_digest_or("x"), "abcdef0123456789");
        let none = AppliedDelta::new_python_measure("m", None, 0);
        assert_eq!(none.short_digest_or("unnamed"), "unnamed");
    }
}
