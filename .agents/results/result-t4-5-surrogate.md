# QA Review: T4.5 Surrogate Model Benchmarking

**Issue**: #720
**Status**: WARNING
**Date**: 2026-05-16
**Reviewer**: qa-reviewer (glm-5.1)

## Executive Summary

The ONNX surrogate model infrastructure is **architecturally complete** but the actual ONNX inference path is **not benchmarkable in its current state**. When `model_loaded == false` (the default for `SurrogateManager::new()`), all inference calls silently fall back to a mock `vec![1.2; n]` response. No benchmark exists that loads a real ONNX model and times inference against the physics baseline. The existing `benchmark_8760_timesteps` and `batch_oracle_bench` benches use the **mock/analytical** path only.

The acceptance criterion ("ONNX inference timing vs physics baseline documented") **cannot be satisfied** without a trained surrogate model and a benchmark that loads it.

---

## Files Reviewed

| File | Purpose |
|------|---------|
| `src/ai/surrogate.rs` | Main surrogate manager: ONNX session pool, inference, fallback |
| `src/ai/modular_surrogate.rs` | Composite/multi-component surrogate composition |
| `src/ai/distributed.rs` | Multi-GPU distributed inference manager |
| `src/ai/shared_batch_service.rs` | Batched inference service with worker thread |
| `benches/benchmark_8760_timesteps.rs` | 8760-timestep benchmark (uses mock path) |
| `benches/batch_oracle_bench.rs` | BatchOracle throughput benchmark |
| `benches/performance.rs` | General performance benchmark |
| `benches/baseline/phase10/BASELINE_SUMMARY.md` | Phase 10 performance baselines |
| `tests/test_batched_inference.rs` | Batched inference correctness tests |
| `tests/test_modular_surrogates.rs` | Modular surrogate tests |
| `tests/surrogate_config.rs` | Surrogate configuration tests |
| `Cargo.toml` | Feature flags, dependencies, bench declarations |

---

## Acceptance Criteria Checklist

| Criterion | Status | Notes |
|-----------|--------|-------|
| ONNX inference timing vs physics baseline documented | **NOT MET** | No benchmark loads a real ONNX model; mock path returns `1.2` values |
| Speedup ratio quantified | **NOT MET** | Cannot measure without real inference |
| Benchmark stub for future use | **PARTIAL** | Existing `benchmark_8760_timesteps` bench has structure but only uses analytical mode |

---

## Review Result: WARNING

### CRITICAL

(None)

### HIGH

- `src/ai/surrogate.rs:727` — **`panic!` on tensor creation failure** in `predict_loads`. A malformed input or version-skewed ONNX model will crash the entire process instead of returning an error. This is unreachable in the current mock path but will be triggered once a real model is loaded.

  ```rust
  // Current (line 726-728):
  Err(e) => {
      panic!("Failed to create input tensor: {}. Error: {}", e, e);
  }

  // Remediation:
  Err(e) => {
      warn!("Failed to create input tensor: {}, using mock loads", e);
      return vec![1.2; current_temps.len()];
  }
  ```

- `src/ai/surrogate.rs:746` — **`panic!("ONNX inference returned no outputs.")`** after a successful session run but empty output. Same issue: should fall back gracefully.

  ```rust
  // Remediation: Replace all panic! calls in predict_loads (lines 746, 749, 754, 758)
  // with warn! + fallback to vec![1.2; current_temps.len()]
  // to match the existing fallback pattern used elsewhere in the codebase.
  ```

- `src/ai/surrogate.rs:781` — **`panic!` on inconsistent batch sizes** in `predict_loads_batched`. Should return `Result<Vec<Vec<f64>>, String>` or silently pad/truncate.

  ```rust
  // Remediation:
  if t.len() != input_size {
      warn!("Inconsistent input sizes in batch: expected {}, found {}", input_size, t.len());
      return batch_temps.iter().map(|temps| vec![1.2; temps.len()]).collect();
  }
  ```

- `benches/benchmark_8760_timesteps.rs:28-35` — **Benchmark uses `SurrogateManager::new()` which creates a mock-only manager** (`model_loaded: false`). This means all surrogate timing data measures only the fallback path returning `vec![1.2]`, not actual ONNX inference.

  ```rust
  // Current:
  let surrogates = SurrogateManager::new().expect("Failed to create SurrogateManager");

  // Remediation (add a conditional ONNX benchmark):
  let surrogates = if std::path::Path::new("models/loads_predictor.onnx").exists() {
      SurrogateManager::load_onnx("models/loads_predictor.onnx")
          .expect("Failed to load ONNX model")
  } else {
      eprintln!("WARNING: No ONNX model found, using analytical fallback");
      SurrogateManager::new().unwrap()
  };
  ```

### MEDIUM

- `src/ai/surrogate.rs:551-570` — **`analytical_loads()` uses `SystemTime::now()` as input**. The analytical fallback incorporates the current wall-clock hour into the solar gain calculation. This makes the function non-deterministic for benchmarking — repeated calls with the same temperature inputs produce different outputs depending on time of day.

  ```rust
  // Current (line 558):
  let hour_of_day = (std::time::SystemTime::now()
      .duration_since(std::time::UNIX_EPOCH)
      .unwrap()
      .as_secs() / 3600) as usize % 24;

  // Remediation: Accept hour_of_day as a parameter with a default method:
  pub fn analytical_loads_at(&self, temps: &[f64], hour_of_day: usize) -> Result<Vec<f64>, String> {
      // ... deterministic version
  }
  pub fn analytical_loads(&self, temps: &[f64]) -> Result<Vec<f64>, String> {
      let hour = /* current hour */;
      self.analytical_loads_at(temps, hour)
  }
  ```

- `benches/batch_oracle_bench.rs` — **No surrogate-mode ONNX benchmark**. The bench calls `oracle.evaluate_population(pop, true)` (surrogate mode) but the underlying manager is mock-only, so it measures analytical-not-ONNX throughput.

- `src/ai/surrogate.rs:770` — **Silent mock fallback without logging** in `predict_loads_batched`. When `model_loaded` is false, it returns `1.2` values with no `warn!` call, unlike `predict_loads_with_fallback` which logs. Inconsistent error handling makes debugging harder.

  ```rust
  // Remediation:
  if !self.model_loaded || batch_temps.is_empty() {
      if !batch_temps.is_empty() {
          warn!("Model not loaded, returning mock loads for batch of {}", batch_temps.len());
      }
      return batch_temps.iter().map(|temps| vec![1.2; temps.len()]).collect();
  }
  ```

- `models/rl_policy/policy.onnx`, `assets/loads_predictor.onnx`, `examples/dummy_surrogate.onnx` — **Three ONNX models exist in the repo** but none are wired into any benchmark. `loads_predictor.onnx` would be the natural candidate for a surrogate-vs-physics timing comparison.

### LOW

- `src/ai/surrogate.rs` — `InferenceMetrics::record_inference()` exists (line ~195) but is **never called** from `predict_loads` or `predict_loads_batched`. The metrics infrastructure is dead code.

- `benches/baseline/phase10/BASELINE_SUMMARY.md` — All BatchOracle surrogate-mode entries show "TBD". Baseline was never populated with surrogate timing data.

- `tests/test_modular_surrogates.rs` — Holdout accuracy tests (`test_holdout_accuracy_solar_component`, `test_holdout_accuracy_hvac_component`) are `#[ignore]` and reference model paths that may not exist.

---

## Architecture Assessment

### What Exists

1. **`SurrogateManager`** — Full ONNX Runtime integration via `ort` crate
   - Session pool with thread-safe access (`parking_lot::Mutex`)
   - Multi-backend support: CPU, CUDA, CoreML, DirectML, OpenVINO
   - GPU feature flag: `cuda = ["ort/cuda", "ort/tensorrt"]`
   - Quantization configs: FP32, FP16, INT8
   - Multi-device session pools
   - Batched inference (`predict_loads_batched`)
   - Fallback to analytical/mock

2. **`CompositeSurrogate`** — Multi-component architecture
   - Component surrogates (solar, HVAC, infiltration, thermal mass)
   - Weighted prediction aggregation
   - Domain validation for input bounds

3. **`SharedBatchInferenceService`** — Production-grade batch service
   - Worker thread with crossbeam channels
   - Dynamic batch sizing (max 512, 10ms wait)
   - Non-blocking submit API

4. **`DistributedSurrogateManager`** — Multi-GPU inference
   - Rayon-based parallel evaluation
   - Device queue management

5. **Criterion benchmarks** — 10 benchmark targets defined in Cargo.toml

### What's Missing

1. **No benchmark that loads a real ONNX model** — All benchmarks use `SurrogateManager::new()` (mock path)
2. **No timing comparison** between `predict_loads` (ONNX) and `solve_timesteps` (physics)
3. **No trained surrogate model** validated against physics output
4. **No benchmark stub** specifically for surrogate-vs-physics comparison

### Theoretical Speedup Estimate

Based on the architecture:

| Component | Physics (per timestep) | Surrogate (ONNX CPU) | Estimated Speedup |
|-----------|----------------------|---------------------|-------------------|
| Thermal solve (1 zone) | ~0.22 µs (from Phase 10 baseline) | ~50-200 µs (ORT overhead) | **0.001-0.004x** (slower!) |
| Thermal solve (100 zones) | ~22 µs (estimated) | ~50-200 µs (same single forward pass) | **0.1-0.4x** |
| CFD/ray-tracing (if replaced) | ~10-1000 ms | ~50-200 µs | **50-20,000x** |

**Key insight**: The surrogate is designed to replace expensive CFD/ray-tracing simulations (seconds to minutes), NOT the already-fast analytical thermal model (~microseconds). For the current Fluxion use case (analytical thermal model), the ONNX surrogate overhead would make inference **slower**, not faster.

---

## Recommended Benchmark Stub

```rust
// benches/surrogate_physics_comparison.rs
//! Benchmark: ONNX surrogate inference vs physics simulation timing.
//! Run: cargo bench --release --bench surrogate_physics_comparison

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use fluxion::ai::surrogate::SurrogateManager;
use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;

const MODEL_PATH: &str = "assets/loads_predictor.onnx";

fn bench_surrogate_vs_physics(c: &mut Criterion) {
    let has_model = std::path::Path::new(MODEL_PATH).exists();

    let physics_model = ThermalModel::<VectorField>::new(1);
    let surrogate = if has_model {
        SurrogateManager::load_onnx(MODEL_PATH).expect("Failed to load ONNX model")
    } else {
        eprintln!("WARNING: No ONNX model at {}, using mock", MODEL_PATH);
        SurrogateManager::new().unwrap()
    };

    let mut group = c.benchmark_group("surrogate_vs_physics");

    for &n_configs in &[1, 10, 100, 1000] {
        group.bench_with_input(BenchmarkId::new("physics", n_configs), &n_configs, |b, &n| {
            b.iter(|| {
                for _ in 0..n {
                    let mut model = physics_model.clone();
                    model.solve_timesteps(
                        black_box(1), &surrogate, black_box(false),
                        black_box(None), black_box(None), black_box(None)
                    );
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("surrogate", n_configs), &n_configs, |b, &n| {
            let temps: Vec<f64> = vec![20.0; 10];
            b.iter(|| {
                for _ in 0..n {
                    black_box(surrogate.predict_loads(&temps));
                }
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_surrogate_vs_physics);
criterion_main!(benches);
```

---

## Verdict

**WARNING** — Zero CRITICAL issues, but 4 HIGH and 4 MEDIUM findings.

The surrogate infrastructure is well-designed with production features (session pooling, multi-GPU, batched inference, quantization). However:
1. No real ONNX benchmark exists — acceptance criterion **not met**
2. Four `panic!` calls in inference paths will crash production servers
3. Analytical fallback is non-deterministic (time-dependent)
4. The speedup target only makes sense for CFD/ray-tracing replacement, not the existing analytical thermal model
