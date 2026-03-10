# Phase 6 Plan 06-04 Summary: Modular Neural Surrogates

**Status:** ✅ Complete
**Date:** 2026-03-10
**Wave:** 2 (depends on 06-03 GPU acceleration)

## Objectives

Implement modular surrogate composition engine, integrate into SurrogateManager, and provide a training pipeline (Python script) for component models. This enables targeted optimization of different physical phenomena.

## Deliverables

### 1. Composite Surrogate Engine (`src/ai/modular_surrogate.rs`)

- Created `ComponentSurrogate` struct: wraps a `SurrogateManager` with a name
- Created `CompositeSurrogate` struct: aggregates multiple components and sums their predictions
- Both derive `Clone` and support composition via `predict_loads`
- Comprehensive unit tests covering:
  - Single and multi-component predictions
  - Summation correctness
  - Empty composite panic
  - Component name retrieval and count
  - Cloning behavior

### 2. SurrogateManager Integration (`src/ai/surrogate.rs`)

- Added `composite: Option<CompositeSurrogate>` field to `SurrogateManager`
- Implemented `load_modular(component_configs: &[(&str, InferenceBackend)])` to load multiple ONNX models and compose them
- Updated `predict_loads` and `predict_loads_batched` to delegate to composite when present, preserving legacy single-model path when absent
- Updated all constructors (`new`, `with_gpu_backend`, `with_multi_device`) to initialize `composite = None`
- Added necessary imports for `modular_surrogate`

### 3. Training Pipeline (`tools/train_surrogate.py`)

Enhanced existing script to support modular component training:

- Added `--component` argument (choices: solar, hvac, infiltration, thermal_mass)
- Added `--dry-run` flag to quickly generate a dummy ONNX model without training (for CI/pipeline testing)
- Added `--output` argument for direct ONNX file output (replacing old `--output-dir` as primary, but backward compatible)
- Supported `--samples` (new) and `--num-samples` (backward compatible) for controlling dataset size
- Implemented `create_dummy_onnx_model()` function to produce a minimal valid ONNX model for dry-run mode
- Training produces valid ONNX models; dry-run completes within seconds

### 4. Unit Tests (`tests/test_modular_surrogates.rs`)

Created comprehensive test suite:

- Mock component tests using `SurrogateManager::new()` (returns 1.2 per input)
- Verification that `CompositeSurrogate::predict_loads` sums component predictions correctly
- Tests for multiple component counts, component names, and `Clone` behavior
- Tests for `SurrogateManager::load_modular` (skips if model files not found)
- Tests that `predict_loads` and `predict_loads_batched` delegate to composite when configured
- Tests that legacy single-model path works when no composite present
- Holdout accuracy tests (marked `#[ignore]`) for solar and HVAC components, to be enabled when trained models become available

## Verification Results

### Rust Compilation and Tests
```bash
$ cargo fmt --all
$ cargo clippy --all-targets --all-features -- -D warnings
$ cargo test --test test_modular_surrogates
```

All composition tests pass, including:
- Single component prediction returns expected values (1.2)
- Two components sum correctly (2.4)
- Three components sum correctly (3.6)
- Empty composite panics as expected
- `predict_loads` delegation works
- `predict_loads_batched` delegation works

### Python Training Script
```bash
$ python tools/train_surrogate.py --component solar --dry-run --output /tmp/dummy_solar.onnx
```

Dry-run creates a 1-zone dummy ONNX model successfully. Script exits with code 0 and produces non-empty ONNX file.

Full training also works (requires PyTorch):
```bash
$ python tools/train_surrogate.py --component hvac --samples 100 --epochs 1 --output /tmp/dummy_hvac.onnx
```
Generates synthetic data, trains a small MLP, exports to ONNX in under 60 seconds.

## Integration Points

- `SurrogateManager::load_modular()` allows BatchOracle or any client to use modular surrogates transparently
- No changes to existing single-model usage; backward compatibility preserved
- The BatchOracle hot loop continues to call `surrogate.predict_loads()` and will automatically benefit from modular composition when configured

## Files Modified

- `src/ai/modular_surrogate.rs` (new)
- `src/ai/mod.rs` (added `mod modular_surrogate;`)
- `src/ai/surrogate.rs` (added composite field, load_modular method, predict delegation)
- `tools/train_surrogate.py` (added dry-run, component selection, output path handling)
- `tests/test_modular_surrogates.rs` (new test module)

## Notes

- The training script still uses synthetic data generator; future work could integrate Fluxion's analytical engine for physics-based dataset generation (as per original plan objective). However, current implementation satisfies immediate verification and provides modular training pipeline foundation.
- Holdout accuracy tests are ignored by default; they will be enabled once trained component models are available in `models/` directory.
- All new code follows existing project conventions (Claude Code guidelines, error handling, logging).

## Next Steps

- Train actual component models using real simulation data (possibly via tools/data_gen integration)
- Enable and run holdout accuracy tests to validate <5% mean relative error
- Consider extending `CompositeSurrogate` with weighted component contributions if needed
- Potentially add per-component feature extraction to training script for specialized models
