# Phase 6 Plan 03 Summary: GPU-Accelerated Surrogate Inference

**Completed:** 2026-03-10
**Status:** ✅ Complete

## Objectives

- Add GPU autodetection respecting `FLUXION_GPU` env var
- Create SharedBatchInferenceService to maximize GPU utilization
- Integrate GPU path into BatchOracle.evaluate_population
- Ensure thread safety

## Deliverables

### 1. GPU Autodetection (SurrogateManager::gpu_supported)

- Added method `gpu_supported()` to `SurrogateManager` (src/ai/surrogate.rs)
- Returns `true` iff:
  - Compiled with `cuda` feature
  - Backend is `InferenceBackend::CUDA`
  - Environment variable `FLUXION_GPU` not set to "0" or "false"
- Unit tests verify behavior for CPU backend and env override (tests/test_gpu_autodetect.rs)

### 2. SharedBatchInferenceService

- Created new module `src/ai/shared_batch_service.rs`
- Exposes:
  - `DynamicBatchConfig`: batch size and wait time
  - `SharedBatchInferenceService`: thread-safe, cloneable service
- Workers submit inference requests via `submit(temps: Vec<f64>) -> Receiver<Vec<f64>>`
- Service aggregates requests into batches and calls `predict_loads_batched`
- Uses `std::sync::mpsc` channels for request queue and oneshot responses
- Automatic shutdown when all senders are dropped
- Comprehensive tests for single request, concurrency, batching, and shutdown (tests/test_shared_batch_service.rs)

### 3. BatchOracle Integration

- Modified `BatchOracle::evaluate_population` (src/lib.rs)
- When `use_surrogates` is true:
  - Checks `self.surrogates.gpu_supported()`
  - If GPU available: creates `SharedBatchInferenceService` and workers use it
  - If GPU not available: retains existing coordinator-worker channel pattern (CPU path)
- GPU path uses same time-first loop, but workers directly submit to service
- No data races: service uses single consumer thread; senders are cloned

## Testing

- `cargo test --test test_gpu_autodetect` — 2 passed
- `cargo test --test test_shared_batch_service` — 4 passed
- Existing CPU tests continue to work (manual verification)

## Performance Notes

- GPU path batches all worker requests per timestep with configurable batch size (set to number of workers)
- Service thread handles aggregation without blocking the main thread
- minimal overhead compared to previous coordinator pattern

## Next Steps

- Validate on hardware with CUDA-capable GPU and ONNX model
- Tune `DynamicBatchConfig` (max_batch_size, wait_ms) for specific hardware
- Consider exposing service reuse across multiple `evaluate_population` calls
