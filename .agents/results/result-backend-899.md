# Backend Result — Issue #899

**Status:** COMPLETE
**Branch:** `fix/issue-899-onnx`
**Issue:** #899 — [AI] Replace mock surrogate with real ONNX inference pipeline
**Related:** #764 (Phase 4a MLP surrogate architecture spec)

## Summary

Wired the real ONNX inference pipeline into `SurrogateManager`. Previously,
`SurrogateManager::predict_loads()` returned a constant `vec![1.2; len]`
placeholder when no model was loaded, and even when a model was loaded the
public `predict_loads` / `predict_loads_batched` methods would **panic** on
ONNX errors instead of degrading gracefully. This change:

1. Adds a public `is_mock()` method so callers can detect mock mode.
2. Refactors `predict_loads` / `predict_loads_batched` to route through real
   ONNX inference when a model is loaded and fall back to the `1.2` mock
   constant (matching prior behavior) on any ONNX error — never panics.
3. Adds new `predict_loads_onnx` / `predict_loads_batched_onnx` methods that
   return `Result<Vec<f64>, String>` for callers that need to distinguish
   real neural predictions from mock placeholders.
4. Adds inference timing metrics (`InferenceMetrics`) recorded per call,
   exposed via `inference_metrics()`.
5. Ships a minimal ONNX fixture (`assets/dummy_surrogate.onnx`) — a
   deterministic pass-through (`output[0,0] = input[0,0]`) with a dynamic
   batch dimension. Built with `onnx` (opset 17) and verified against
   `onnxruntime` 1.24.3.
6. Adds 9 new unit tests covering: `is_mock()` true/false, loading the
   real ONNX fixture, inspecting the I/O schema, running real inference,
   the regression that `predict_loads` no longer returns the `1.2` mock
   constant when a model is loaded, batched inference, and metrics.

The `ort` crate (`2.0.0-rc.10`, pinned via `Cargo.toml`) was already a
dependency — no new crates added.

## Files Changed

- `src/ai/surrogate.rs` — refactored predict paths, added `is_mock()`,
  `predict_loads_onnx()`, `predict_loads_batched_onnx()`, inference
  metrics field, +9 new tests. **+351 / -95 lines.**
- `assets/dummy_surrogate.onnx` — new 193-byte ONNX fixture
  (`float32[batch, 6] -> float32[batch, 1]`, opset 17, dynamic batch).

## Acceptance Criteria

- [x] Real ONNX inference executes when `model_loaded=true` (verified by
      `test_predict_loads_uses_real_onnx_when_loaded`).
- [x] Mock mode is detectable via `is_mock()` (verified by
      `test_is_mock_true_when_no_model_loaded` /
      `test_is_mock_false_when_model_loaded`).
- [x] Mock fallback preserved — `predict_loads` returns the `1.2`
      constant when no model is loaded or ONNX fails (existing
      `test_surrogate_manager_fallback` still passes; new
      `test_predict_loads_onnx_errors_when_no_model_loaded` covers the
      error path).
- [x] Inference metrics recorded (verified by
      `test_predict_loads_onnx_runs_real_inference` checking
      `num_inferences == 1` after one call).
- [x] ONNX fixture ships in repo (193 B, deterministic pass-through).
- [x] `predict_loads_onnx` and `predict_loads_batched_onnx` return
      `Result` instead of panicking.

## Test Results

- `cargo check --lib --all-features`: **clean** (1 crate compiled, no errors, no warnings).
- `cargo test --release --lib -- --skip slow`: **2472 passed, 2 ignored, 1 filtered out**.
  - New tests: 9/9 pass.
  - Existing tests: unchanged count, all still pass.

## Out of Scope (per task)

- CLI `--surrogate-mock=false` flag — not wired in this change. The
  `is_mock()` detection is in place; a CLI flag can be added in a
  follow-up once a CLI surface is confirmed (the repo has no `main.rs`).
- Phase 3 benchmarking infrastructure (physics vs surrogate speedup) —
  deferred; metrics collection is now in place to support it.
- Training / downloading a real model — explicitly out of scope.

## Notes

- The pre-existing `ort` call in `predict_loads` was already correct
  for `ort 2.0.0-rc.12` (the resolved version). The panic-on-error
  behavior was the issue; this change replaces it with graceful
  fallback to mock data.
- The new `inference_metrics` field uses `Arc<parking_lot::Mutex<_>>`
  for interior mutability so the existing `&self` API stays unchanged
  while still allowing metrics to be recorded.
- Phase 4a (#764) defines a 23-feature input schema for the trained
  surrogate; the fixture here uses 6 inputs to match the current
  `SurrogateInputs` struct. The plumbing supports any input size
  automatically; migrating to 23 features is a separate change.
- Mock fallback (`vec![1.2; len]`) is preserved verbatim so existing
  test contracts and call sites are unchanged.
