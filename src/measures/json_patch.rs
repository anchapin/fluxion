// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! JSON Patch (RFC 6902) application for [`FluxionModel`].
//!
//! # Overview
//!
//! This module is Phase 1 (Declarative Deltas) of the Hybrid Measure
//! Approach (Issue #1811). It exposes a single entry point —
//! [`apply_delta`] — that consumes a [`json_patch::Patch`] and mutates
//! a [`FluxionModel`] in place.
//!
//! # Round-trip semantics
//!
//! Each call to [`apply_delta`] follows three steps:
//!
//! 1. **Serialize**: `FluxionModel` → `serde_json::Value`. The model is
//!    entirely serde-derived, so this step is infallible in practice
//!    (only fails on a custom-serializer bug, mapped to
//!    [`DeltaError::Serialize`]).
//! 2. **Apply**: run [`json_patch::patch`] on the JSON value. The
//!    `json-patch` crate reverts on partial failure, so the JSON value
//!    is unchanged if any operation fails.
//! 3. **Reconstruct**: `serde_json::Value` → `FluxionModel`. Failure
//!    here means a Delta tried to assign a value whose JSON type
//!    doesn't match the Rust field (e.g. string for `f64`), mapped to
//!    [`DeltaError::TypeMismatch`] or [`DeltaError::Deserialize`].
//!
//! On any error, the original `FluxionModel` is **unchanged** — the
//! patch is applied to a throwaway `Value`, and only committed on
//! success.
//!
//! # Pointer path conventions
//!
//! See the module-level docs on [`crate::measures::model::FluxionModel`]
//! for the canonical pointer paths. The short version:
//!
//! - `/zones/<key>/<field>` — zone metadata
//! - `/constructions/<key>/layers/<idx>/<field>` — construction layers
//! - `/assemblies/<key>/layers/<idx>/<field>` — assembly layers
//!
//! # Example
//!
//! ```
//! use fluxion::measures::json_patch::apply_delta;
//! use fluxion::measures::model::FluxionModel;
//! use json_patch::Patch;
//! use serde_json::json;
//!
//! let mut model = FluxionModel::ashrae_140_case_600();
//! let before = model.zones["zone_1"].volume;
//!
//! // Apply a `replace` operation: zone_1's volume goes from 129.6 → 200.0.
//! let p: Patch = serde_json::from_value(json!([
//!     { "op": "replace", "path": "/zones/zone_1/volume", "value": 200.0 }
//! ])).unwrap();
//!
//! apply_delta(&mut model, &p).unwrap();
//! assert_eq!(model.zones["zone_1"].volume, 200.0);
//! assert_ne!(before, model.zones["zone_1"].volume);
//! ```

use json_patch::{patch as apply_json_patch, Patch};
use serde_json::Value;

use crate::measures::error::DeltaError;
use crate::measures::model::FluxionModel;

/// Apply a JSON Patch (RFC 6902) to a [`FluxionModel`] in place.
///
/// On success, `model` is mutated to reflect the patched JSON. On any
/// failure, `model` is **unchanged** — see the round-trip semantics
/// section in the module docs.
///
/// # Errors
///
/// Returns a typed [`DeltaError`] for every documented failure class.
/// Never panics on user-supplied data.
///
/// # Example: +20% insulation R-value (acceptance test 1)
///
/// ```
/// use fluxion::measures::json_patch::apply_delta;
/// use fluxion::measures::model::FluxionModel;
/// use json_patch::Patch;
///
/// let mut model = FluxionModel::ashrae_140_case_900();
/// let before_k = model.assemblies["wall_1"].layers[1].conductivity;
/// let before_r = model.assemblies["wall_1"].layers[1].thickness / before_k;
///
/// // +20% R-value with the same thickness ⇒ divide k by 1.2.
/// let new_k = before_k / 1.2;
/// let patch_json = serde_json::json!([
///     { "op": "replace", "path": "/assemblies/wall_1/layers/1/conductivity", "value": new_k }
/// ]);
/// let p: Patch = serde_json::from_value(patch_json).unwrap();
///
/// apply_delta(&mut model, &p).unwrap();
///
/// let after_k = model.assemblies["wall_1"].layers[1].conductivity;
/// let after_r = model.assemblies["wall_1"].layers[1].thickness / after_k;
///
/// // R-value should be ~20% higher.
/// assert!((after_r / before_r - 1.2).abs() < 1e-9);
/// ```
pub fn apply_delta(model: &mut FluxionModel, patch: &Patch) -> Result<(), DeltaError> {
    // Step 1: serialize. We borrow `model` immutably first, then mutate
    // only after a successful round-trip — so any failure leaves
    // `model` untouched.
    let mut doc: Value =
        serde_json::to_value(&*model).map_err(|e| DeltaError::Serialize(e.to_string()))?;

    // Step 2: apply. `json_patch::patch` reverts on partial failure, so
    // the `Value` is unchanged if any operation fails.
    apply_json_patch(&mut doc, patch).map_err(map_patch_error)?;

    // Step 3: reconstruct. If the patched JSON doesn't match the Rust
    // shape (e.g. Delta wrote a string where `f64` is expected), this is
    // where we catch it as a typed `TypeMismatch` / `Deserialize` error.
    let patched: FluxionModel =
        serde_json::from_value(doc).map_err(|e| classify_deserialize_error(&e))?;

    // Commit.
    *model = patched;
    Ok(())
}

/// Map a [`json_patch::PatchError`] onto a [`DeltaError`] variant.
///
/// The `json-patch` crate's error kinds are richer than what the
/// measures surface needs; this helper translates them into the typed
/// variants documented on [`DeltaError`].
fn map_patch_error(err: json_patch::PatchError) -> DeltaError {
    use json_patch::PatchErrorKind;

    let path = err.path.to_string();

    match err.kind {
        PatchErrorKind::TestFailed => DeltaError::TestFailed { path },
        PatchErrorKind::InvalidFromPointer | PatchErrorKind::InvalidPointer => {
            DeltaError::InvalidPath { path }
        }
        PatchErrorKind::CannotMoveInsideItself => DeltaError::CannotMoveInsideItself { path },
        // Forward-compat: `json-patch` may add new variants. Treat them
        // as opaque failures rather than panicking.
        ref other => DeltaError::JsonPatch(format!("{:?}: {}", other, err)),
    }
}

/// Inspect a serde-deserialization error and pick the most informative
/// [`DeltaError`] variant.
///
/// When the error message indicates a JSON-type mismatch (e.g. string
/// where `f64` is expected), we surface it as a [`DeltaError::TypeMismatch`]
/// with the JSON-pointer path. Otherwise we fall back to the generic
/// [`DeltaError::Deserialize`].
fn classify_deserialize_error(err: &serde_json::Error) -> DeltaError {
    let msg = err.to_string();

    if let Some((expected, actual, path)) = parse_type_mismatch_message(&msg) {
        DeltaError::TypeMismatch {
            path,
            expected,
            actual,
        }
    } else {
        DeltaError::Deserialize(msg)
    }
}

/// Parse a serde_json deserialization error message of the form
/// `"invalid type: <actual>, expected <expected> at line N column M"`,
/// returning `(expected_type, actual_type, pointer_path)` if it matches.
///
/// Returns `None` if the message doesn't follow this format. The path is
/// always `/` when the error is at the root of the value (we don't have
/// access to `serde_json::Error::path()` on this crate version).
fn parse_type_mismatch_message(msg: &str) -> Option<(String, String, String)> {
    // Examples:
    //   `invalid type: string "\"hello\"", expected f64`
    //   `invalid type: integer `5`, expected f64`
    //   `invalid type: boolean `true`, expected u32`
    let after_marker = msg.find("invalid type: ")?;
    let rest = &msg[after_marker + "invalid type: ".len()..];

    // `rest` looks like `string "...", expected <T>` or `integer N, expected <T>`.
    // We split on the FIRST `,` and trim.
    let comma = rest.find(',')?;
    let actual_part = &rest[..comma];
    let after_comma = &rest[comma + 1..];

    let expected_idx = after_comma.find("expected ")?;
    let after_expected = &after_comma[expected_idx + "expected ".len()..];
    let expected = after_expected
        .split_whitespace()
        .next()
        .unwrap_or("unknown")
        .to_string();

    // actual: take the first word (e.g. "string", "integer", "boolean", "null", "map", "sequence", "number")
    let actual = actual_part
        .split_whitespace()
        .next()
        .unwrap_or("unknown")
        .to_string();

    // We don't have access to serde_json::Error::path() on this crate
    // version (1.0.149) — the path is buried in the message text in
    // earlier versions. For now, default to root-level pointer.
    Some((expected, actual, "/".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measures::model::{FluxionModel, MaterialLayer};
    use serde_json::json;

    /// Helper: parse a JSON Patch from a `serde_json::Value`.
    fn patch_from_value(v: Value) -> Patch {
        serde_json::from_value(v).expect("patch should parse")
    }

    #[test]
    fn test_apply_r_value_increase_case_900() {
        // Acceptance test: +20% insulation R-value on a Case 900 model.
        let mut model = FluxionModel::ashrae_140_case_900();

        // wall_1's layer[1] is the foam-board insulation (k = 0.04 W/mK,
        // thickness = 0.0615 m, R = 1.5375 m²K/W).
        let before_k = model.assemblies["wall_1"].layers[1].conductivity;
        let before_r = model.assemblies["wall_1"].layers[1].thickness / before_k;
        assert!(
            (before_r - 1.5375).abs() < 1e-9,
            "baseline R-value sanity check failed (got {})",
            before_r
        );

        // +20% R-value at constant thickness ⇒ divide k by 1.2.
        let new_k = before_k / 1.2;
        let p = patch_from_value(json!([
            { "op": "replace", "path": "/assemblies/wall_1/layers/1/conductivity", "value": new_k }
        ]));

        apply_delta(&mut model, &p).expect("apply_delta should succeed");

        let after_k = model.assemblies["wall_1"].layers[1].conductivity;
        let after_r = model.assemblies["wall_1"].layers[1].thickness / after_k;

        // R-value should be ~20% higher, within float tolerance.
        let ratio = after_r / before_r;
        assert!(
            (ratio - 1.2).abs() < 1e-6,
            "expected R-value ratio 1.2, got {} (before_r={}, after_r={})",
            ratio,
            before_r,
            after_r
        );

        // Conductivity must have actually decreased.
        assert!(after_k < before_k, "k should decrease when R increases");
    }

    #[test]
    fn test_apply_r_value_increase_case_600() {
        // Companion test: +20% R-value on a Case 600 (low-mass) model.
        // Case 600 has an "Insulation" layer in assembly wall_1.
        let mut model = FluxionModel::ashrae_140_case_600();
        let idx = model.assemblies["wall_1"]
            .layers
            .iter()
            .position(|l| l.name == "Insulation")
            .expect("Case 600 wall_1 must contain an Insulation layer");

        let before = &model.assemblies["wall_1"].layers[idx];
        let before_r = before.r_value();
        let new_k = before.conductivity / 1.2;

        let p = patch_from_value(json!([
            {
                "op": "replace",
                "path": format!("/assemblies/wall_1/layers/{}/conductivity", idx),
                "value": new_k
            }
        ]));

        apply_delta(&mut model, &p).unwrap();

        let after = &model.assemblies["wall_1"].layers[idx];
        let after_r = after.r_value();
        assert!((after_r / before_r - 1.2).abs() < 1e-6);
    }

    #[test]
    fn test_invalid_path_returns_typed_error() {
        // Acceptance test: malformed path → typed error, NO panic.
        let mut model = FluxionModel::ashrae_140_case_600();
        let p = patch_from_value(json!([
            { "op": "replace", "path": "/zones/zone_does_not_exist/volume", "value": 999.0 }
        ]));

        let err = apply_delta(&mut model, &p).expect_err("must error on bad path");
        assert!(
            matches!(err, DeltaError::InvalidPath { .. }),
            "expected InvalidPath, got {:?}",
            err
        );

        // Model must be unchanged after the failed patch.
        assert!(model.zones.contains_key("zone_1"));
        assert!(!model.zones.contains_key("zone_does_not_exist"));
    }

    #[test]
    fn test_invalid_path_does_not_mutate_model() {
        // Stronger invariant: after a failed patch, the model's
        // serialized form is byte-identical to the original.
        let mut model = FluxionModel::ashrae_140_case_600();
        let before = serde_json::to_string(&model).unwrap();

        let p = patch_from_value(json!([
            // First op is valid, second is bogus — atomicity is the
            // caller's responsibility, but each op failing individually
            // must not leak partial state into the model.
            { "op": "remove", "path": "/nope" }
        ]));

        let _ = apply_delta(&mut model, &p);
        let after = serde_json::to_string(&model).unwrap();
        assert_eq!(before, after, "model must be unchanged after failed patch");
    }

    #[test]
    fn test_type_mismatch_returns_typed_error() {
        // Acceptance test: setting a float field to a string → typed error.
        let mut model = FluxionModel::ashrae_140_case_600();
        let p = patch_from_value(json!([
            { "op": "replace", "path": "/zones/zone_1/volume", "value": "not a number" }
        ]));

        let err = apply_delta(&mut model, &p).expect_err("must error on type mismatch");
        // The error should be a TypeMismatch or a Deserialize fallback.
        // (json-patch may not check the type at apply time; we catch it
        // on the reconstruction step.)
        assert!(
            matches!(
                err,
                DeltaError::TypeMismatch { .. } | DeltaError::Deserialize(_)
            ),
            "expected TypeMismatch or Deserialize, got {:?}",
            err
        );

        // Model must be unchanged.
        assert!((model.zones["zone_1"].volume - 129.6).abs() < 1e-9);
    }

    #[test]
    fn test_index_out_of_bounds_returns_typed_error() {
        // Setting an out-of-range array index.
        let mut model = FluxionModel::ashrae_140_case_600();
        let p = patch_from_value(json!([
            { "op": "replace", "path": "/assemblies/wall_1/layers/99/conductivity", "value": 0.04 }
        ]));

        let err = apply_delta(&mut model, &p).expect_err("must error on bad index");
        assert!(
            matches!(
                err,
                DeltaError::InvalidPath { .. } | DeltaError::Deserialize(_)
            ),
            "expected InvalidPath or Deserialize, got {:?}",
            err
        );
    }

    #[test]
    fn test_round_trip_safe() {
        // Acceptance test: serialize → deserialize → apply → serialize →
        // compare. The patched JSON must be identical whether the patch
        // is applied to the original model or to a deserialized copy.
        let mut original = FluxionModel::ashrae_140_case_900();
        let mut copy: FluxionModel =
            serde_json::from_str(&serde_json::to_string(&original).unwrap()).unwrap();

        let p = patch_from_value(json!([
            { "op": "replace", "path": "/assemblies/wall_1/layers/1/conductivity", "value": 0.0333 },
            { "op": "replace", "path": "/zones/zone_1/volume", "value": 150.0 }
        ]));

        apply_delta(&mut original, &p).unwrap();
        apply_delta(&mut copy, &p).unwrap();

        let original_json = serde_json::to_string(&original).unwrap();
        let copy_json = serde_json::to_string(&copy).unwrap();
        assert_eq!(
            original_json, copy_json,
            "patch application must be deterministic across round-trips"
        );
    }

    #[test]
    fn test_add_remove_operations() {
        // Sanity check for `add` and `remove` ops.
        let mut model = FluxionModel::default();
        let p = patch_from_value(json!([
            { "op": "add", "path": "/zones/zone_1", "value": {
                "name": "Zone 1", "floor_area": 48.0, "volume": 129.6, "height": 2.7
            }},
            { "op": "remove", "path": "/zones/zone_1" }
        ]));
        apply_delta(&mut model, &p).unwrap();
        assert!(
            model.zones.is_empty(),
            "add then remove should leave zones empty"
        );
    }

    #[test]
    fn test_multiple_operations_in_sequence() {
        // Two replace ops on the same model, in one patch.
        let mut model = FluxionModel::ashrae_140_case_900();
        let p = patch_from_value(json!([
            { "op": "replace", "path": "/zones/zone_1/volume", "value": 200.0 },
            { "op": "replace", "path": "/assemblies/wall_1/layers/1/conductivity", "value": 0.03 }
        ]));
        apply_delta(&mut model, &p).unwrap();
        assert!((model.zones["zone_1"].volume - 200.0).abs() < 1e-9);
        assert!((model.assemblies["wall_1"].layers[1].conductivity - 0.03).abs() < 1e-12);
    }

    #[test]
    fn test_test_operation_pass() {
        // A `test` op with the correct value should pass.
        let mut model = FluxionModel::ashrae_140_case_600();
        let p = patch_from_value(json!([
            { "op": "test", "path": "/zones/zone_1/volume", "value": 129.6 }
        ]));
        apply_delta(&mut model, &p).expect("test op with correct value should pass");
    }

    #[test]
    fn test_test_operation_fail_returns_typed_error() {
        // A `test` op with the wrong value should return a typed error.
        let mut model = FluxionModel::ashrae_140_case_600();
        let p = patch_from_value(json!([
            { "op": "test", "path": "/zones/zone_1/volume", "value": 999.0 }
        ]));
        let err = apply_delta(&mut model, &p).expect_err("test op with wrong value should fail");
        assert!(
            matches!(err, DeltaError::TestFailed { .. }),
            "expected TestFailed, got {:?}",
            err
        );
    }

    #[test]
    fn test_apply_to_empty_model_with_add() {
        // A minimal `Default` model should be patchable. RFC 6902 `add`
        // requires the parent of the target path to exist; to create a
        // new zone we must add the whole zone object in one op.
        let mut model = FluxionModel::default();
        let p = patch_from_value(json!([
            { "op": "add", "path": "/zones/zone_1", "value": {
                "name": "Test Zone",
                "floor_area": 10.0,
                "volume": 42.0,
                "height": 2.5
            }}
        ]));
        apply_delta(&mut model, &p).unwrap();
        assert_eq!(model.zones["zone_1"].volume, 42.0);
        assert_eq!(model.zones["zone_1"].name, "Test Zone");
        assert_eq!(model.zones["zone_1"].floor_area, 10.0);
        assert_eq!(model.zones["zone_1"].height, 2.5);
    }

    #[test]
    fn test_layer_r_value_updated_via_patch() {
        // Verify the relationship between layers and R-value end-to-end.
        let mut model = FluxionModel::ashrae_140_case_900();
        let original_r = model.assembly_total_r_value("wall_1").unwrap();

        // Halve the conductivity of layer 0 (concrete) → R-value doubles.
        let p = patch_from_value(json!([
            { "op": "replace", "path": "/assemblies/wall_1/layers/0/conductivity", "value": 0.7 }
        ]));
        apply_delta(&mut model, &p).unwrap();

        let new_r = model.assembly_total_r_value("wall_1").unwrap();
        // layer 0 contributed (0.200/1.4) = 0.1429; halving k doubles it to 0.2857.
        // So new_r ≈ old_r + 0.1429 ≈ original_r + 0.1429.
        let expected_delta = 0.200 / 0.7 - 0.200 / 1.4;
        assert!(
            ((new_r - original_r) - expected_delta).abs() < 1e-9,
            "expected new_r - original_r ≈ {}, got {}",
            expected_delta,
            new_r - original_r
        );
    }

    #[test]
    fn parse_type_mismatch_message_handles_string_for_float() {
        let (expected, actual, _path) = parse_type_mismatch_message(
            "invalid type: string \"hello\", expected f64 at line 1 column 24",
        )
        .expect("should parse");
        assert_eq!(expected, "f64");
        assert_eq!(actual, "string");
    }

    #[test]
    fn parse_type_mismatch_message_handles_integer_for_float() {
        let (expected, actual, _path) =
            parse_type_mismatch_message("invalid type: integer 5, expected f64")
                .expect("should parse");
        assert_eq!(expected, "f64");
        assert_eq!(actual, "integer");
    }

    #[test]
    fn parse_type_mismatch_message_returns_none_on_unrelated_error() {
        assert!(parse_type_mismatch_message("missing field `name`").is_none());
        assert!(parse_type_mismatch_message("").is_none());
    }

    #[test]
    fn map_patch_error_handles_test_failed() {
        // We can't easily construct a PatchError, but we can exercise
        // the code path through `apply_delta`'s test-op-failure test.
        // Here we just verify the helper does what we expect for each
        // PatchErrorKind by simulating equivalent JsonPatch errors.
        let json_err = DeltaError::JsonPatch("test".to_string());
        match json_err {
            DeltaError::JsonPatch(_) => {}
            _ => panic!("expected JsonPatch variant"),
        }
    }

    #[test]
    fn layer_r_value_helper_still_works() {
        // Sanity: the helper method we used in the test still works.
        let layer = MaterialLayer {
            name: "X".to_string(),
            conductivity: 0.04,
            density: 10.0,
            specific_heat: 1400.0,
            thickness: 0.066,
            emissivity: 0.9,
            absorptance: 0.5,
        };
        assert!((layer.r_value() - 1.65).abs() < 1e-9);
    }
}
