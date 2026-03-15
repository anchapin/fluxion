# Phase 6: Performance Optimization - Context

**Gathered:** 2026-03-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Optimize batch validation throughput to achieve <5 minute execution time for all 18+ ASHRAE 140 cases and add GPU-accelerated calculations.

**Key Requirements:**
- GPU acceleration for neural surrogates (CUDA backend)
- Batch inference optimization with dynamic batching
- Parallel execution of validation cases using rayon
- Performance regression guardrails (MAE, throughput, execution time)
- Historical performance tracking and trend analysis

**Out of scope:**
- GPU acceleration of solar calculations (deferred to future phase)
- Multi-node distributed computing (beyond multi-GPU)
- Real-time interactive visualization (Phase 7)

</domain>

---

<decisions>
## Implementation Decisions

### GPU Acceleration Strategy

**Primary target:** Surrogates on GPU first (100x speedup driver)

- Enable CUDA backend for SurrogateManager with auto-detection
- Use ONNX Runtime's CUDAExecutionProvider with session pool
- Keep solar calculations on CPU (analytical) for now — separate follow-up phase

**Activation policy:** Auto-detect when GPU available, with configurable override

- Runtime check: `cuda` feature enabled AND CUDA drivers present
- Config option: `validation.gpu.enabled = true/false` (default: auto)
- Environment variable: `FLUXION_GPU=1` forces GPU, `FLUXION_GPU=0` forces CPU
- CLI flag: `--gpu` / `--no-gpu` overrides config

**Multi-GPU support:** Optional — support both single and multi-GPU

- Single GPU: `gpu_devices: [0]` → SurrogateManager with one device
- Multi-GPU: `gpu_devices: [0, 1, 2]` → DistributedSurrogateManager with round-robin load balancing
- Config: `validation.gpu.devices = [usize]` (default: `[0]` if GPU present)
- Graceful fallback: if specified devices not available, warn and use CPU

**Memory management:** Adaptive batching with existing `DynamicBatchConfig`

- Use `max_batch_size` to limit GPU memory usage (default: 512)
- Respect `min_batch_size` and `max_wait_ms` for latency/throughput trade-off
- No custom memory pool — rely on ONNX Runtime's internal management
- Monitor GPU memory via `peak_memory_mb` metric; recommend users tune `max_batch_size` if OOM

---

### Batch Validation Parallelization

**Granularity:** Hybrid — case-level parallelism + population-level within each case

- Main thread spawns rayon `par_iter()` over the list of ASHRAE cases (18+ items)
- Each case independently calls `BatchOracle.evaluate_population()` with its parameter vector
- Each `evaluate_population()` already uses rayon internally for population parallelism
- This is safe: each case's population parallelism runs in its own rayon scope, not nested

**Work distribution:** Dynamic work stealing via rayon native scheduler

- No manual grouping or chunking — just `cases.par_iter().map(|case| run_case(case))`
- Rayon automatically balances based on case complexity (1-zone vs 2-zone)
- Granularity is coarse (18 cases) so overhead negligible

**GPU coordination:** Batch across cases with async aggregation

- **New architecture needed:** All cases submit inference requests to a central batcher
- Batcher collects requests from all parallel case workers, forms large batches, runs single GPU inference
- Returns results to respective workers
- Implementation: `SharedBatchInferenceService` with thread-safe request queue
- Trade-off: adds async complexity but maximizes GPU utilization (fewer small kernel launches)

**Progress monitoring:** Context-aware — aggregate summary for CI, progress bar for local

- Detect CI via `CI` environment variable or `--ci` flag
- CI mode: Print per-case completion line `[600] ✓ 1.2s` to stderr, final summary to stdout
- Local mode: Use `indicatif` progress bar showing `X/Y cases` with ETA
- Structured JSON log written to `target/validation_results.jsonl` regardless of mode

---

### Performance Regression Guardrails

**Integration:** Hybrid — lightweight guardrails in validation suite + separate benchmark binary

- `fluxion validate` records metrics, compares to baseline, prints warnings/errors
- `fluxion benchmark` runs detailed microbenchmarks (batch sizes, kernel profiling) for manual analysis
- Validation suite guardrails don't require extra binary; just compute metrics and compare

**Regression detection:** Threshold-based for CI, plus historical storage for trend analysis

- Accuracy regressions (physics correctness):
  - MAE increase >2% (relative to baseline) → **fail CI**
  - Max Deviation increase >10% → **fail CI**
  - Pass rate drop >5 percentage points → **fail CI**
- Performance regressions (execution time):
  - Total validation time >110% of baseline → **warn only** (hardware-sensitive)
  - Throughput drop >20% → **warn only**

**Historical data storage:** JSON append (`target/performance_history.jsonl`)

- Each validation run appends one JSON object:
  ```json
  {
    "timestamp": "2026-03-10T14:38:00Z",
    "git_sha": "abc123",
    "mae": 12.3,
    "max_deviation": 18.2,
    "pass_rate": 91.7,
    "validation_time_seconds": 245.1,
    "throughput_configs_per_sec": 12345
  }
  ```
- Not committed to git (in `target/`), avoids merge conflicts
- Can be parsed line-by-line for trend plots

**Alert severity:**
- Accuracy regressions → fail CI (blocks merge, physics correctness is paramount)
- Performance regressions → warning in CI output (posted as PR comment), build passes
- Always print human-readable comparison: `MAE: 12.3% (baseline 10.5%) ↑+17% ❌`

---

### Surrogate Training & Validation Strategy

**Training data source:** Fluxion physics-generated data, with spot checking against ASHRAE reference

- Generate dataset by running `BatchOracle.evaluate_population(use_surrogates=false)` across parameter space
- Parameter ranges: U-value (0.1-5.0), heating setpoint (15-25), cooling setpoint (22-32), plus additional design variables as needed
- Capture input temperatures (8760 × num_zones) and output loads (8760 × num_zones) as training pairs
- Spot check: Compare Fluxion analytical results against ASHRAE reference values for key cases (900, 960) to ensure ground truth is reasonable

**Training scope:** Modular surrogates — separate model per component

- Components:
  - `solar_gains_surrogate.onnx` — Perez sky model + window SHGC
  - `hvac_loads_surrogate.onnx` — HVAC demand calculation
  - `infiltration_surrogate.onnx` — Air change rate effects
  - `thermal_mass_surrogate.onnx` — Mass dynamics (optional, may not need)
- Each surrogate takes relevant inputs (temperatures, weather, geometry) and predicts component loads
- Composed at runtime: total load = sum(surrogate predictions) + analytical residuals
- Benefits: Easier to train (smaller networks), isolate failures, replace individual components

**Validation protocol:** Holdout set testing (±5%) + case-by-case validation (±15%)

- Holdout set: 20% of training data not seen during training
  - Requirement: surrogate predictions within ±5% of analytical ground truth on holdout set
  - Test both RMSE and R² (>0.98 recommended)
- Case-by-case: After training, run each ASHRAE 140 case with surrogates, compare annual energy to analytical
  - Requirement: each case's annual heating/cooling within ±15% of analytical result
  - If any case fails, surrogate not used for that case (fallback to analytical)
- Validation runs automatically during `maturin build` or `cargo test --release` if ONNX model present

**Fallback behavior:** Automatic fallback with sanity check

- Before running full year with surrogate, run first N timesteps (e.g., 100 hours) with both surrogate and analytical
- Compute relative difference: `|surrogate - analytical| / max(analytical, 1e-6)`
- If mean difference >10% over sanity check window, fall back to analytical for entire run
- Log clear warning: `Surrogate validation failed (12% error), falling back to analytical`
- Users can override with `surrogate.strict = false` to use surrogate anyway (not recommended)

**Surrogate scope:** All thermal loads but keep modular

- Target all load components: solar, infiltration, convection, HVAC
- Implement each as separate ONNX model for modularity
- Integration point: `SurrogateManager::predict_loads_batched()` composes modular predictions
- Future: can replace individual modules without retraining entire system

---

### Claude's Discretion

- Exact threshold values for guardrails (2% MAE, 10% MaxDev, 110% time) — can be tuned based on first baseline runs
- Spot check methodology against ASHRAE reference (which cases, how many samples)
- Surrogate network architectures (layer counts, widths, activation functions)
- Sanity check window size (N=100 timesteps) and tolerance (10%)
- Modular surrogate breakdown — which components deserve separate models vs. combined

</decisions>

---

<code_context>
## Existing Code Insights

### Reusable Assets

**BatchOracle infrastructure:**
- `BatchOracle::evaluate_population()` already implements population-level parallelism with rayon
- Parameter validation and model cloning pattern established
- Time-first loop architecture for surrogate usage (already designed for GPU)

**Surrogate management:**
- `SurrogateManager` with `SessionPool` for concurrent ONNX inference (src/ai/surrogate.rs)
- `DistributedSurrogateManager` for multi-GPU setups (src/ai/distributed.rs)
- Support for CUDA, OpenVINO, CoreML, DirectML backends already in code
- `predict_loads_batched()` method exists for batch inference

**Dynamic batching:**
- `DynamicBatchManager` and `BatchProcessor` with configurable batch sizes (src/ai/batch_inference.rs)
- `DynamicBatchConfig` with presets for low-latency and high-throughput
- Benchmark utilities for measuring batch performance

**Validation framework:**
- `ASHRAE140Validator` with tolerance-based pass/fail (src/validation/)
- `BenchmarkReport` aggregates results across cases
- `ValidationReportGenerator` produces Markdown summaries
- Diagnostic logging and CSV export already in place (Phase 5)

**Parallel execution:**
- rayon used throughout codebase (engine, validation, batch inference)
- `rayon::prelude::ParallelIterator` pattern established
- Thread pool configuration via `RAYON_NUM_THREADS` env var

### Established Patterns

**Two-class API pattern:**
- `BatchOracle` for high-throughput population evaluation (optimization loops)
- `Model` for detailed single-building analysis (validation/inspection)
- Both use `ThermalModel<VectorField>` internally with same physics

**Continuous Tensor Abstraction (CTA):**
- All state variables are `VectorField` (element-wise operations)
- Enables future GPU acceleration of physics calculations
- Batched operations on VectorField already support rayon parallelism

**Configuration via builder pattern:**
- `ThermalModelBuilder` for constructing models from `CaseSpec`
- Case specifications include geometry, constructions, HVAC, weather
- Patterns for extending with new fields (door_geometry, etc.)

**Validation-driven development:**
- Every phase includes test scaffolds before implementation
- Holdout set testing pattern from Phase 2-3 thermal mass validation
- Case-specific benchmarks with reference ranges

### Integration Points

**Where new code connects:**

- `src/ai/surrogate.rs` — Extend `SurrogateManager` to support modular ONNX models, compose predictions
- `src/ai/batch_inference.rs` — Add `SharedBatchInferenceService` for cross-case async batching
- `src/lib.rs` (`BatchOracle::evaluate_population`) — Modify surrogate path to use new batched aggregation when GPU enabled
- `src/validation/ashrae_140_validator.rs` — Add performance metric recording (time, throughput) to `BenchmarkReport`
- `src/validation/reporter.rs` — Extend `ValidationReportGenerator` to include performance summary section
- `src/bin/fluxion.rs` — Add CLI flags: `--gpu`, `--no-gpu`, `--ci`, and config file support
- `Cargo.toml` — Ensure `ort` crate has `CUDA` feature enabled for GPU builds

**New files to create:**
- `tools/train_surrogate.py` — Data generation and training pipeline for modular surrogates
- `tests/surrogate_validation.rs` — Holdout set testing and case-by-case accuracy checks
- `src/ai/modular_surrogate.rs` — Composition engine for modular surrogate predictions
- `src/ai/shared_batch_service.rs` — Async request aggregation for multi-case GPU batching

**Configuration schema (new `config/validation.yaml` or similar):**
```yaml
validation:
  gpu:
    enabled: auto  # auto|true|false
    devices: [0]   # list of GPU device IDs
  batch:
    max_size: 512
    min_size: 16
    wait_ms: 10
  guardrails:
    fail_on_accuracy_regression: true
    mae_threshold: 0.02      # 2%
    max_deviation_threshold: 0.10  # 10%
    time_warn_factor: 1.10   # 110%
  surrogate:
    enabled: false  # off by default until fully validated
    sanity_check: true
    sanity_window: 100  # timesteps
    sanity_tolerance: 0.10  # 10%
```

</code_context>

---

<specifics>
## Specific Ideas

### Performance Baseline Establishment

Before implementing guardrails, establish baseline metrics on reference hardware:

1. Run `fluxion validate --all` on clean checkout (no changes) 3 times
2. Record: mean execution time, MAE, Max Deviation, Pass Rate
3. Store in `docs/performance_baseline.json`:
   ```json
   {
     "timestamp": "2026-03-10",
     "git_sha": "HEAD",
     "hardware": "8-core CPU, RTX 4090",
     "mae": 10.5,
     "max_deviation": 15.2,
     "pass_rate": 100.0,
     "validation_time_seconds": 245.1
   }
   ```
4. Guardrails compare against these values, not against previous CI run

**Rationale:** Hardware varies across CI runners; baseline should be hardware-specific but stable reference.

---

### GPU Autodetection Logic

Implement autodetection in `SurrogateManager::new()`:

```rust
pub fn new() -> Result<Self> {
    let cuda_available = cfg!(feature = "cuda") && std::env::var("FLUXION_GPU").map(|v| v == "1").unwrap_or(true);
    if cuda_available {
        // Try CUDA backend, fall back to CPU if fails
        Self::with_gpu_backend("models/surrogate.onnx", InferenceBackend::CUDA, 0)
            .or_else(|_| Self::new())  // fall back to CPU mock
    } else {
        Self::new()  // CPU mode
    }
}
```

**Note:** `cfg!(feature = "cuda")` checks if compiled with CUDA support. Runtime check for driver availability via `ort::session::Session::new()`.

---

### Async Batch Aggregation Architecture

Design `SharedBatchInferenceService`:

```rust
pub struct SharedBatchInferenceService {
    surrogate: Arc<SurrogateManager>,
    request_queue: Arc<SegQueue<InferenceRequest>>,
    batch_config: DynamicBatchConfig,
    worker_thread: Option<JoinHandle<()>>,
}

impl SharedBatchInferenceService {
    pub fn new(surrogate: SurrogateManager, config: DynamicBatchConfig) -> Self {
        let queue = Arc::new(SegQueue::new());
        let service = SharedBatchInferenceService { surrogate: Arc::new(surrogate), queue: Arc::clone(&queue), batch_config: config, worker_thread: None };
        service.start_worker();
        service
    }

    pub fn submit(&self, inputs: &[Vec<f64>]) -> oneshot::Sender<Vec<Vec<f64>>> {
        let (tx, rx) = oneshot::channel();
        self.queue.push(InferenceRequest { inputs: inputs.to_vec(), response_tx: tx });
        rx
    }

    fn start_worker(&self) {
        let queue = Arc::clone(&self.queue);
        let surrogate = Arc::clone(&self.surrogate);
        self.worker_thread = Some(std::thread::spawn(move || {
            loop {
                // Collect batch
                let mut batch = Vec::new();
                let mut senders = Vec::new();
                while let Ok(req) = queue.try_pop() {
                    batch.push(req.inputs);
                    senders.push(req.response_tx);
                    if batch.len() >= self.batch_config.max_batch_size {
                        break;
                    }
                }
                if batch.is_empty() {
                    std::thread::yield_now();
                    continue;
                }
                // Flatten batch (batch of batches → single large batch)
                let flat_batch: Vec<Vec<f64>> = batch.into_iter().flatten().collect();
                let results = surrogate.predict_loads_batched(&flat_batch);
                // Distribute results back to requesters
                // ... (split flat results back to original batch grouping)
            }
        }));
    }
}
```

**Integration:** `BatchOracle::evaluate_population()` with `use_surrogates=true` creates a `SharedBatchInferenceService` at the start, has all worker threads submit requests to it, collects results at end.

---

### Holdout Set Testing Implementation

In `tests/surrogate_validation.rs`:

```rust
#[test]
fn test_surrogate_holdout_accuracy() {
    // Load pre-generated dataset (temperatures → loads)
    let (train_data, holdout_data) = load_surrogate_dataset().split(0.8);

    // Train surrogate (invoked externally via Python script)
    // train_surrogate(&train_data);

    // Load trained ONNX model
    let surrogate = SurrogateManager::load_onnx("models/surrogate.onnx").unwrap();

    // Evaluate on holdout
    let mut errors = Vec::new();
    for (input, expected_output) in holdout_data {
        let predicted = surrogate.predict_loads(&input);
        let error = (predicted - expected_output).abs();  // or relative error %
        errors.push(error);
    }

    let mean_relative_error = errors.iter().map(|e| e.mean()).sum::<f64>() / errors.len() as f64;
    assert!(mean_relative_error < 0.05, "Holdout error {:.2}% exceeds 5% threshold", mean_relative_error * 100.0);
}
```

**Note:** Actual training happens outside Rust (Python with PyTorch/TensorFlow). Rust tests only validate trained model.

---

### Baseline Comparison Output Format

When validation runs with guardrails enabled, output should look like:

```
✅ Validation complete (18/18 cases passed)

Performance Metrics:
  MAE: 11.2% (baseline 10.5%) ↑+7% ⚠️  (threshold: >2% fail)
  Max Deviation: 16.8% (baseline 15.2%) ↑+10% ⚠️  (threshold: >10% warn)
  Pass Rate: 100.0% (baseline 100.0%) ✓
  Validation time: 248.3s (baseline 245.1s) ↑+1% ✓

⚠️  WARNING: Max Deviation increased by 10% (threshold: 10%)
See: target/performance_history.jsonl
```

If accuracy regression exceeds threshold:
```
❌ VALIDATION FAILED: MAE increased by 17% (threshold: 2%)
Performance regression does not fail the build, but accuracy regression does.
```

---

</specifics>

---

<deferred>
## Deferred Ideas

None identified — Phase 6 scope focused on performance optimization as discussed. All ideas captured within phase boundaries.

**Future phases that may address related needs:**
- Phase 7 (Advanced Analysis) may add visualization of performance trends
- Separating GPU acceleration for solar calculations could be a Phase 7 or 8 enhancement

</deferred>

---

*Phase: 06-performance-optimization*
*Context gathered: 2026-03-10*
