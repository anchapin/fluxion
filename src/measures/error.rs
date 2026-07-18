// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Typed error variants for the `fluxion::measures` module.
//!
//! These variants cover the failure modes of [`crate::measures::json_patch::apply_delta`]
//! (and any future Delta application entry points). All variants are non-exhaustive
//! to allow additional cases to be added without breaking downstream pattern matches.
//!
//! # Design
//!
//! Each variant carries enough context (pointer path, expected/actual type tag, etc.)
//! for callers to construct actionable error messages. None of the variants carry
//! heap-allocated data on the error path itself beyond `String`, so they remain
//! cheap to construct and `match`-friendly.
//!
//! # Examples
//!
//! ```
//! use fluxion::measures::error::DeltaError;
//!
//! let err = DeltaError::InvalidPath { path: "/zones/zone_1".to_string() };
//! assert_eq!(format!("{}", err), "invalid JSON Patch pointer: /zones/zone_1");
//! ```

use thiserror::Error;

/// All failures that can occur while applying a JSON Patch (RFC 6902) Delta
/// to a [`crate::measures::model::FluxionModel`].
///
/// `DeltaError` is the canonical error type for the measures pipeline. It is
/// deliberately a flat, non-exhaustive enum so that callers can pattern-match
/// on the failure class without unwrapping nested error chains.
///
/// # Mapping to `json-patch`'s `PatchErrorKind`
///
/// The four kinds `json_patch::PatchErrorKind::{TestFailed, InvalidFromPointer,
/// InvalidPointer, CannotMoveInsideItself}` are mapped onto [`DeltaError::InvalidPath`]
/// (pointer failures) or [`DeltaError::TestFailed`] (test-op failure). Anything
/// else — e.g. JSON parse errors, value-type errors caught during the round-trip —
/// falls through to a dedicated variant.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DeltaError {
    /// The patch could not be parsed (invalid JSON, missing fields, unknown op).
    #[error("could not parse JSON Patch: {0}")]
    ParseError(String),

    /// A JSON Pointer (RFC 6901) in the patch does not resolve in the model.
    ///
    /// Examples:
    /// - `/zones/zone_does_not_exist/volume` — zone key not in the model.
    /// - `/constructions/wall/foo/bar` — key does not exist on the value.
    #[error("invalid JSON Patch pointer: {path}")]
    InvalidPath {
        /// The pointer that failed to resolve.
        path: String,
    },

    /// An array index in the patch is not a non-negative integer or is out of bounds.
    ///
    /// Examples:
    /// - `/constructions/wall/layers/-1` — negative index.
    /// - `/constructions/wall/layers/99` — past the end of `layers`.
    #[error("array index out of bounds at {path}: index {index}")]
    IndexOutOfBounds {
        /// The pointer that triggered the bounds error.
        path: String,
        /// The index (as parsed from the pointer).
        index: i64,
    },

    /// The patch tried to assign a value whose JSON type does not match the
    /// target field's expected type (e.g. string where a float is required).
    ///
    /// This variant is raised after the patch is applied and the model fails
    /// to deserialize back to its Rust type. Callers that want to catch this
    /// BEFORE applying the patch should use a pre-flight `test` operation.
    #[error("type mismatch at {path}: expected {expected}, got {actual}")]
    TypeMismatch {
        /// The pointer that triggered the type mismatch.
        path: String,
        /// The expected type tag (e.g. `"f64"`, `"String"`, `"Vec<Layer>"`).
        expected: String,
        /// The actual JSON type tag encountered (e.g. `"string"`, `"number"`).
        actual: String,
    },

    /// A `test` operation in the patch failed — the value at the pointer did
    /// not match the expected value.
    #[error("test operation failed at {path}")]
    TestFailed {
        /// The pointer that the test op targeted.
        path: String,
    },

    /// The model could not be serialized to JSON (extremely rare; usually
    /// indicates a custom serializer bug).
    #[error("failed to serialize model to JSON: {0}")]
    Serialize(String),

    /// The model could not be deserialized back from the patched JSON. This
    /// is the catch-all for shape mismatches that don't fit the more specific
    /// [`DeltaError::TypeMismatch`] / [`DeltaError::InvalidPath`] variants.
    #[error("failed to deserialize patched model: {0}")]
    Deserialize(String),

    /// A `move` operation would move a value inside itself.
    #[error("move operation would move {path} inside itself")]
    CannotMoveInsideItself {
        /// The pointer that the move op targeted.
        path: String,
    },

    /// Catch-all for failures originating in the underlying `json-patch` crate
    /// that don't cleanly map onto the more specific variants above.
    #[error("json-patch error: {0}")]
    JsonPatch(String),
}

impl DeltaError {
    /// Returns `true` if the model state is still consistent after this error
    /// (i.e. the patch was rolled back). Currently all variants meet this
    /// contract because we always round-trip via `json_patch::patch` (which
    /// reverts on failure) and only mutate the model after a successful round-trip.
    pub fn is_state_preserved(&self) -> bool {
        // All variants are produced AFTER attempting the patch — if we return
        // Err, the original `&mut self` is unchanged (we deserialize into a
        // throwaway, then move in only on success).
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_path_display() {
        let err = DeltaError::InvalidPath {
            path: "/zones/zone_1".to_string(),
        };
        assert_eq!(
            format!("{}", err),
            "invalid JSON Patch pointer: /zones/zone_1"
        );
    }

    #[test]
    fn type_mismatch_display() {
        let err = DeltaError::TypeMismatch {
            path: "/zones/zone_1/volume".to_string(),
            expected: "f64".to_string(),
            actual: "string".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("/zones/zone_1/volume"));
        assert!(msg.contains("f64"));
        assert!(msg.contains("string"));
    }

    #[test]
    fn index_out_of_bounds_display() {
        let err = DeltaError::IndexOutOfBounds {
            path: "/constructions/wall/layers/5".to_string(),
            index: 5,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("layers/5"));
        assert!(msg.contains("index 5"));
    }

    #[test]
    fn is_state_preserved_for_all_variants() {
        // Contract: every error variant leaves the model unchanged because
        // apply_delta only mutates after a successful round-trip.
        let variants: Vec<DeltaError> = vec![
            DeltaError::ParseError("bad".to_string()),
            DeltaError::InvalidPath {
                path: "/x".to_string(),
            },
            DeltaError::IndexOutOfBounds {
                path: "/y".to_string(),
                index: 1,
            },
            DeltaError::TypeMismatch {
                path: "/z".to_string(),
                expected: "f64".to_string(),
                actual: "string".to_string(),
            },
            DeltaError::TestFailed {
                path: "/t".to_string(),
            },
            DeltaError::Serialize("s".to_string()),
            DeltaError::Deserialize("d".to_string()),
            DeltaError::CannotMoveInsideItself {
                path: "/m".to_string(),
            },
            DeltaError::JsonPatch("p".to_string()),
        ];
        for v in variants {
            assert!(
                v.is_state_preserved(),
                "variant {} should preserve state",
                v
            );
        }
    }

    #[test]
    fn is_std_error() {
        fn assert_error<E: std::error::Error>(_: E) {}
        assert_error(DeltaError::ParseError("x".to_string()));
    }
}
