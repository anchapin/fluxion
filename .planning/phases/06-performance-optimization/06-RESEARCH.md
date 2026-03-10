# Phase 6: Performance Optimization - Research

**Researched:** 2026-03-10
**Domain:** Rust performance optimization, GPU acceleration, ONNX Runtime, parallel validation
**Confidence:** HIGH

## Summary

Fluxion's Phase 6 targets two major performance improvements: (1) parallelizing ASHRAE 140 validation to achieve <5 minute execution time for all 18 cases, and (2) adding GPU-accelerated neural surrogate inference using ONNX Runtime's CUDA backend. The codebase already has substantial infrastructure: `SurrogateManager` with `SessionPool` for concurrent inference, `DynamicBatchManager` for adaptive batching, and a coordinator-worker pattern in `BatchOracle::evaluate_population` for time-first surrogate execution. Key work involves connecting these components, adding case-level parallelism, implementing performance metrics collection, and establishing regression guardrails with historical tracking.

The existing validation framework (`ASHRAE140Validator::validate_analytical_engine`) runs cases sequentially and lacks any performance monitoring. This is the primary bottleneck. We'll modify it to use rayon's `par_iter()` for case-level parallelism while preserving the existing per-case simulation logic. For GPU acceleration, we'll extend `SurrogateManager` with GPU autodetection and integrate it into `BatchOracle` so that when `use_surrogates=true` with GPU enabled, all cases share a centralized batched inference service that maximizes GPU utilization across parallel workers.

**Primary recommendation:** Implement hybrid parallelization (case-level + population-level) with a shared GPU batching layer, and augment `BenchmarkReport` with performance metrics for guardrail enforcement.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **GPU Acceleration Strategy:** Primary target is surrogates on GPU first (100x speedup driver). Enable CUDA backend for SurrogateManager with auto-detection. Keep solar calculations on CPU (analytical) for now — separate follow-up phase.
- **Activation policy:** Auto-detect when GPU available, with configurable override via config file (`validation.gpu.enabled = true/false`), environment variable (`FLUXION_GPU=1/0`), and CLI flag (`--gpu`/`--no-gpu`).
- **Multi-GPU support:** Optional — support both single and multi-GPU with round-robin load balancing. Config: `validation.gpu.devices = [usize]`.
- **Memory management:** Adaptive batching with existing `DynamicBatchConfig`. Use `max_batch_size` to limit GPU memory usage (default: 512). No custom memory pool — rely on ONNX Runtime's internal management.
- **Batch Validation Parallelization:** Hybrid — case-level parallelism + population-level within each case. Main thread spawns rayon `par_iter()` over ASHRAE cases (18+ items). Each case independently calls `BatchOracle.evaluate_population()`. Each `evaluate_population()` already uses rayon internally for population parallelism. This is safe: each case's population parallelism runs in its own rayon scope, not nested.
- **GPU coordination:** Batch across cases with async aggregation. New architecture needed: All cases submit inference requests to a central batcher. Batcher collects requests from all parallel case workers, forms large batches, runs single GPU inference. Returns results to respective workers. Implementation: `SharedBatchInferenceService` with thread-safe request queue.
- **Progress monitoring:** Context-aware — aggregate summary for CI, progress bar for local. Detect CI via `CI` environment variable or `--ci` flag. CI mode prints per-case completion lines to stderr. Local mode uses `indicatif` progress bar. Structured JSON log written to `target/validation_results.jsonl` regardless of mode.
- **Performance Regression Guardrails:** Hybrid — lightweight guardrails in validation suite + separate benchmark binary. Accuracy regressions fail CI: MAE increase >2% (relative to baseline), Max Deviation increase >10%, Pass rate drop >5 percentage points. Performance regressions warn only: Total validation time >110% of baseline, Throughput drop >20%.
- **Historical data storage:** JSON append to `target/performance_history.jsonl` with fields: timestamp, git_sha, mae, max_deviation, pass_rate, validation_time_seconds, throughput_configs_per_sec.
- **Surrogate Training & Validation Strategy:** Training data from Fluxion physics-generated data. Modular surrogates: separate ONNX model per component (solar_gains, hvac_loads, infiltration, thermal_mass). Holdout set testing (±5%) + case-by-case validation (±15%). Automatic fallback with sanity check: run first 100 timesteps with both surrogate and analytical; if mean difference >10%, fall back to analytical.
- **Surrogate scope:** All thermal loads but keep modular. Composed at runtime in `SurrogateManager::predict_loads_batched()`.
- **Embedding approach:** Fourier basis neural representation (`NeuralScalarField`) for continuous field modeling.

### Claude's Discretion

- Exact threshold values for guardrails (2% MAE, 10% MaxDev, 110% time) — can be tuned based on first baseline runs
- Spot check methodology against ASHRAE reference (which cases, how many samples)
- Surrogate network architectures (layer counts, widths, activation functions)
- Sanity check window size (N=100 timesteps) and tolerance (10%)
- Modular surrogate breakdown — which components deserve separate models vs. combined

### Deferred Ideas (OUT OF SCOPE)

- GPU acceleration of solar calculations (deferred to future phase)
- Multi-node distributed computing (beyond multi-GPU)
- Real-time interactive visualization (Phase 7)
</user_constraints>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| GPU-01 | ONNX Runtime integrated with CUDA backend for parallel solar calculations | Existing `SurrogateManager::with_gpu_backend()` uses CUDAExecutionProvider; GPU autodetection via `MultiDeviceSessionPool::detect_cuda_devices()`; batch inference with `predict_loads_batched()` |
| GPU-02 | Batch inference optimization for neural surrogates with GPU kernel acceleration | `DynamicBatchManager` provides adaptive batching; coordinator-worker pattern in `BatchOracle` enables time-first batching; need `SharedBatchInferenceService` for cross-case aggregation |
| GPU-03 | GPU memory management for large population evaluations | `DynamicBatchConfig.max_batch_size` controls GPU memory; ONNX Runtime internal memory pool; monitor `peak_memory_mb` metric via `InferenceMetrics` |
| SURR-01 | ONNX Runtime session pool for concurrent AI surrogate inference | `SessionPool` already implemented with guard pattern; `MultiDeviceSessionPool` for multi-GPU; tests confirm pooled sessions work |
| SURR-02 | Batched surrogate inference with rayon for population-level parallelism | `predict_loads_batched()` exists; `BatchOracle::evaluate_population()` uses coordinator-worker pattern; need to extend to case-level parallelism |
| SURR-03 | Neural surrogates trained and integrated for expensive physics calculations | Existing `SurrogateManager` loads ONNX models via `load_onnx()`; modular composition infrastructure exists; training pipeline to be added (tools/train_surrogate.py) |
| BATCH-01 | All 18+ ASHRAE 140 cases executed in parallel with rayon | `validate_analytical_engine()` is currently sequential; convert to `cases.par_iter().map(|case| run_case(case))` using rayon `ParallelIterator` |
| BATCH-02 | Aggregated validation results collected and summarized automatically | `BenchmarkReport` aggregates results; need to extend to include performance metrics and store JSONL history |
| BATCH-03 | Complete validation suite execution time <5 minutes | Case-level parallelism + GPU surrogates should achieve 5-10x speedup; baseline measurement needed |
| REG-01 | MAE tracked and alert generated when >2% | `BenchmarkReport::mae()` exists; need guardrail comparison against baseline and CI fail logic |
| REG-02 | Max Deviation tracked and alert generated when >10% | `BenchmarkReport::max_deviation()` exists; need guardrail comparison |
| REG-03 | Pass rate trends monitored over time to detect performance regression | `BenchmarkReport::pass_rate()` exists; need historical storage and trend analysis |
| REG-04 | Historical performance data stored for long-term trend analysis | Append JSON to `target/performance_history.jsonl`; must include timestamp, git_sha, metrics |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `rayon` | 1.10 | CPU parallelism for case-level and population-level parallelization | Native Rust work-stealing, no GIL, scales with core count |
| `ort` (ONNX Runtime) | 2.0.0-rc.10 | GPU-accelerated neural inference with CUDA backend | Industry standard for ONNX, mature CUDA integration, session pooling |
| `crossbeam` | 0.8.4 | Channel-based coordination for time-first batching | lock-free channels, integrates with rayon scoped threading |
| `tokio` | 1.40 | Async runtime for `SharedBatchInferenceService` (optional) | If using async batching; else crossbeam sufficient |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `clap` | 4.5 | CLI argument parsing for `--gpu`, `--ci` flags | Extending `src/bin/fluxion.rs` |
| `serde`/`serde_json` | 1.0 | Serialize performance history to JSONL | Historical tracking |
| `chrono` | 0.4 | Timestamp generation for performance records | Historical tracking |
| `indicatif` | (add) | Progress bars for local runs | User experience enhancement |
| `criterion` | 0.5 | Microbenchmarking for `fluxion benchmark` | Performance tuning |
| `tempfile` | 3.10 | Test fixtures for surrogate validation | Testing |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `rayon` | `std::thread` + manual join | More control but complex load balancing; rayon work-stealing is proven |
| `ort` CUDA | `tch-rs` (PyTorch) | PyTorch C++ API is less stable for Rust; ONNX Runtime is deployment-focused |
| `crossbeam` channels | `std::sync::mpsc` | crossbeam faster, supports select, better for high-throughput |
| `tokio` async | pure threads | Async needed only if batching service uses async I/O; pure threads simpler for now |
| `indicatif` | plain println! | indicatif provides ETA, throughput estimates; worth dependency |

**Installation:**
```bash
# Add dependencies to Cargo.toml
cargo add indicatif@0.17.7  # if using progress bars
cargo add serde_json chrono  # already present

# Ensure ort has CUDA support (feature not needed for ort 2.0.0-rc.10 with download-binaries)
# Users need CUDA toolkit installed (11.8+ recommended) and NVIDIA drivers
```

## Architecture Patterns

### Recommended Project Structure
```
src/
├── ai/
│   ├── surrogate.rs          # Extend with GPU autodetection, modular composition
│   ├── batch_inference.rs    # Add SharedBatchInferenceService
│   └── distributed.rs        # MultiDeviceSessionPool already exists
├── validation/
│   ├── ashrae_140_validator.rs  # Add parallel case execution, timing
│   ├── benchmark.rs              # Extend BenchmarkReport with performance metrics
│   ├── reporter.rs               # Update report templates with performance section
│   └── analyzer.rs               # Add guardrail checking
├── bin/
│   ├── fluxion.rs               # Add --gpu, --ci, --no-gpu flags; guardrail output
│   └── fluxion_benchmark.rs     # New: detailed microbenchmarks (optional)
└── lib.rs                       # BatchOracle: integrate SharedBatchInferenceService
```

### Pattern 1: Hybrid Parallel Validation

**What:** Combine rayon case-level parallelism with existing population-level parallelism inside each case, plus centralized GPU batcher across all cases.

**When to use:** `fluxion validate --all` with `use_surrogates=true` and GPU enabled.

**Execution flow:**
```
main thread: cases.par_iter()
  ├─ Case 600 worker thread (rayon scope): 1000 population configs → 8760 timesteps
  │   ├─ Each timestep: send temperatures to SharedBatchInferenceService
  │   └─ Receive loads, continue simulation
  ├─ Case 900 worker thread (rayon scope): 500 population configs → 8760 timesteps
  │   └─ same pattern (different population sizes)
  └─ Case 960 worker thread (rayon scope): 200 population configs → 8760 timesteps
      └─ ...

SharedBatchInferenceService (single background thread):
  └─ Collects requests from all workers, forms optimal batches, runs GPU inference
```

**Why nested rayon is safe:** The outer `par_iter()` creates separate rayon scopes for each case. Inside each scope, we spawn additional rayon tasks for population parallelism but these run within the same scope's thread pool, not creating nested global parallelism. See `BatchOracle::evaluate_population()` lines 663-700 for existing scoped pattern.

**Example:**
```rust
// From src/validation/ashrae_140_validator.rs
pub fn validate_all_parallel(&mut self, use_surrogates: bool, gpu_enabled: bool) -> BenchmarkReport {
    use rayon::prelude::*;
    let cases = self.get_all_cases();
    let weather = Arc::new(DenverTmyWeather::new());

    let results: Vec<BenchmarkReport> = cases
        .par_iter()
        .map(|case| {
            let spec = case.spec();
            let results = self.simulate_case_parallel(&spec, &weather, use_surrogates, gpu_enabled);
            let mut report = BenchmarkReport::new();
            // populate report for this case
            report
        })
        .collect();

    // Merge reports into single BenchmarkReport
    results.into_iter().flatten().collect()
}
```

**Source:** Existing `BatchOracle::evaluate_population()` implements scoped parallelism (src/lib.rs:663-700). rayon documentation on nested parallelism: https://docs.rs/rayon/latest/rayon/ (use `rayon::scope()` to avoid oversubscription).

### Pattern 2: SharedBatchInferenceService with Thread-Safe Queue

**What:** A singleton service that aggregates inference requests from multiple parallel workers into optimally-sized batches before GPU submission.

**When to use:** When running validation with GPU surrogates; prevents each worker from submitting tiny batches (inefficient GPU utilization).

**Implementation:**
```rust
// New file: src/ai/shared_batch_service.rs
pub struct SharedBatchInferenceService {
    surrogate: Arc<SurrogateManager>,
    config: DynamicBatchConfig,
    request_queue: Arc<SegQueue<InferenceRequest>>,  // use crossbeam::SegQueue for lock-free
    batch_worker: JoinHandle<()>,
}

impl SharedBatchInferenceService {
    pub fn new(surrogate: SurrogateManager, config: DynamicBatchConfig) -> Self {
        let queue = Arc::new(SegQueue::new());
        let surrogate = Arc::new(surrogate);
        let worker = {
            let queue = Arc::clone(&queue);
            let surrogate = Arc::clone(&surrogate);
            std::thread::spawn(move || {
                Self::batch_worker_loop(queue, surrogate, config)
            })
        };
        Self { surrogate, config, request_queue: queue, batch_worker: worker }
    }

    pub fn submit(&self, inputs: Vec<f64>) -> oneshot::Sender<Vec<f64>> {
        let (tx, rx) = std::sync::mpsc::sync_channel(0);  // sync channel for backpressure
        self.request_queue.push(InferenceRequest { inputs, response_tx: tx });
        rx
    }

    fn batch_worker_loop(queue: Arc<SegQueue<InferenceRequest>>, surrogate: Arc<SurrogateManager>, config: DynamicBatchConfig) {
        loop {
            let mut batch = Vec::new();
            let mut senders = Vec::new();

            // Drain queue with timeout (adaptive)
            while let Ok(req) = queue.try_pop() {
                batch.push(req.inputs);
                senders.push(req.response_tx);
                if batch.len() >= config.max_batch_size {
                    break;
                }
            }

            if batch.is_empty() {
                std::thread::yield_now();
                continue;
            }

            // Run batched inference
            let results = surrogate.predict_loads_batched(&batch);

            // Send results back
            for (sender, result) in senders.into_iter().zip(results) {
                let _ = sender.send(result);
            }
        }
    }
}
```

**Why not dynamic batching with channels?** The existing `DynamicBatchManager` uses `Mutex<Vec<BatchRequest>>` which can bottleneck under high contention. `SegQueue` is lock-free and scales better across many rayon workers. The `SharedBatchInferenceService` is purpose-built for cross-worker aggregation, whereas `DynamicBatchManager` is for intra-worker batching.

**Integration point:** In `BatchOracle::evaluate_population()` when `use_surrogates=true` and `gpu_enabled=true`, create a `SharedBatchInferenceService` at the start of the rayon scope, pass a clone to each worker, replace `self.surrogates.predict_loads_batched()` with `service.submit(temps).recv()`.

**Trade-off:** This adds a network round-trip per timestep (submission + wait), but the batch aggregation gains (fewer GPU kernel launches, better occupancy) far outweigh the overhead for GPU inference. For CPU inference, stick with the existing coordinator-worker pattern which already batches at the population level.

### Pattern 3: Performance Metrics Collection via Decorator Pattern

**What:** Wrap `ASHRAE140Validator::simulate_case()` with timing instrumentation that records per-case duration, cumulative times, and throughput.

**When to use:** During `validate_analytical_engine()` and any benchmark runs.

**Example:**
```rust
// src/validation/analyzer.rs extension
#[derive(Clone, Debug)]
pub struct PerformanceMetrics {
    pub case_id: String,
    pub simulation_time_secs: f64,
    pub num_timesteps: usize,
    pub throughput_configs_per_sec: f64,  // if population-based
    pub gpu_memory_peak_mb: Option<f64>,
}

impl BenchmarkReport {
    // Add field: pub performance_metrics: HashMap<String, PerformanceMetrics>

    pub fn record_performance(&mut self, case_id: String, time_secs: f64, num_configs: usize) {
        let throughput = num_configs as f64 / time_secs;
        self.performance_metrics.insert(case_id, PerformanceMetrics {
            case_id,
            simulation_time_secs: time_secs,
            num_timesteps: 8760,
            throughput_configs_per_sec: throughput,
            gpu_memory_peak_mb: None,  // TODO: query from SurrogateManager::InferenceMetrics
        });
    }

    pub fn total_validation_time(&self) -> f64 {
        self.performance_metrics.values().map(|m| m.simulation_time_secs).sum()
    }
}
```

**Source:** The `BatchOracle` already has `std::time::Instant` benchmarks in `lib.rs` (lines 993, 1013, 1107, 1120). Pattern copied from there.

### Anti-Patterns to Avoid

- **Nested rayon without scopes:** `par_iter()` inside `par_iter()` without `rayon::scope()` oversubscribes threads, causing contention. Always use scoped spawning for inner parallelism (already done in `BatchOracle`).
- **One GPU session per worker:** If each parallel case creates its own `SurrogateManager`, they'll each allocate GPU memory and serialize access. Solution: shared `SessionPool` or `SharedBatchInferenceService`.
- **Recording all hourly diagnostics in parallel:** CSV export + hourly data storage blows memory. Keep diagnostics opt-in via `DiagnosticConfig`.
- **Blocking the main thread on GPU batching:** The batch worker must run on a dedicated thread; never blocks case workers.
- **Using `Mutex` for batch queue:** Under high contention, mutex becomes bottleneck. Use `crossbeam::SegQueue` or `crossbeam::channel` for lock-free.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Concurrent ONNX sessions | Custom session lifecycle management | `SessionPool` + `SessionGuard` (existing) | Handles session return on drop, prevents leaks |
| Adaptive batching logic | Manual batch-size tuning | `DynamicBatchConfig` with presets (low_latency, high_throughput) | Already implemented, test-driven |
| GPU autodetection | Enumerate CUDA devices manually | `MultiDeviceSessionPool::detect_cuda_devices()` | Uses ONNX Runtime's provider detection |
| Work-stealing scheduler | Custom thread pool | `rayon::par_iter()` automatically balances case workloads | Cases vary in complexity (1-zone vs 2-zone); rayon handles it |
| Regression threshold math | Custom deviation calculations | `BenchmarkReport::mae()`, `max_deviation()`, `pass_rate()` | Already implemented and tested |
| JSON appends with file locking | Manual append coordination | `std::fs::OpenOptions::append()` with `lock` crate if needed (unlikely concurrent access) | Simple; performance history not contended |

**Key insight:** Fluxion already has most infrastructure. Phase 6 is primarily *integration* of existing components (SessionPool, rayon, validation) and adding *monitoring* (performance metrics, guardrails). Do not reinvent session pooling or batching; reuse `SessionPool` and extend `DynamicBatchManager` pattern.

## Common Pitfalls

### Pitfall 1: Deadlock in SharedBatchInferenceService

**What goes wrong:** The batch worker blocks indefinitely on `queue.try_pop()` if using blocking channel and no backpressure handling, or if senders are dropped unexpectedly.

**Why it happens:** The `oneshot::Sender` or `mpsc::Sender` can be dropped if a case worker panics or exits early, leaving the batch worker waiting forever.

**How to avoid:** Use non-blocking `try_pop()` in a loop with `std::thread::yield_now()` when empty. Ensure `InferenceRequest` has owned `Vec<f64>` (clone if needed) so the sender can drop immediately. Set a timeout on receiver using `std::time::Duration::from_millis(config.max_wait_ms)` to force batch processing even if not full.

**Warning signs:** GPU sits idle despite pending requests; CPU usage low; validation hangs at specific case.

### Pitfall 2: GPU Memory OOM with Large Batches

**What goes wrong:** `max_batch_size` set too high (e.g., 2048) causes CUDA out-of-memory on modest GPUs (8GB).

**Why it happens:** ONNX Runtime allocates GPU memory for input/output tensors proportional to batch size × feature size. Surrogate models with 100+ inputs can exhaust memory quickly.

**How to avoid:** Start with conservative `max_batch_size=512` (as per CONTEXT.md). Add runtime OOM detection: catch `ort::Error` with "out of memory" and retry with half batch size. Expose `peak_memory_mb` in `InferenceMetrics` to guide users.

**Warning signs:** Validation fails with "CUDA error 2: out of memory" or onnxruntime error. GPU utilization spikes to 100% then crashes.

### Pitfall 3: Over-subscription of Threads Leading to Contention

**What goes wrong:** Setting `RAYON_NUM_THREADS` to system core count AND spawning many rayon scopes plus batch worker thread can oversubscribe, causing cache thrashing and slowdowns.

**Why it happens:** Case-level `par_iter()` spawns N worker threads (where N = `RAYON_NUM_THREADS`). Each case spawns additional rayon scope with M workers (population parallelism). If N=16 and each case spawns M=4, total threads = 16*4 = 64 on a 16-core machine = 4× oversubscription.

**How to avoid:** Limit population parallelism inside each case to `rayon::current_num_threads() / num_active_cases`. Or simpler: run `validate_all_parallel()` with `rayon::ThreadPoolBuilder::new().num_threads(8).build_global()` to limit total threads. The CONTEXT.md suggests using `RAYON_NUM_THREADS` wisely.

**Warning signs:** CPU utilization >95% but throughput (cases/sec) decreases with more cases.

### Pitfall 4: Nested Parallelism Panic

**What goes wrong:** Rust panics with "cannot initialize second `rayon::ThreadPool`" if attempting to create nested thread pools.

**Why it happens:** Attempting to call `rayon::ThreadPoolBuilder::new().build()` inside an existing rayon parallel iterator creates a nested pool, which rayon forbids.

**How to avoid:** Do not create custom thread pools inside parallel iterators. Use `rayon::scope()` for inner parallelism, which reuses the outer pool's threads. If absolutely need different pool settings, call `rayon::ThreadPoolBuilder::new().build_global()` *before* starting outer `par_iter()`.

**Warning signs:** Panic message: "attempted to initialize a new `rayon::ThreadPool` while already inside a rayon parallel iterator".

### Pitfall 5: Guardrails Blocking CI Due to Hardware Variability

**What goes wrong:** Performance guardrails fail because CI runner is 2× slower than baseline machine, not due to actual regression.

**Why it happens:** Total validation time guardrail compares absolute seconds. Hardware differences (CPU generation, GPU model) cause natural variance.

**How to avoid:** Use *relative* guardrails for performance: throughput (configs/sec) normalized by CPU benchmark score (e.g., `sysbench`). Or only warn, not fail. Accuracy guardrails (MAE, MaxDeviation) are hardware-independent and should fail.

**Warning signs:** Pass rate 100% but CI fails on "validation time >110% baseline". Baseline needs to be per-hardware profile or guardrails need tuning.

## Code Examples

### Example 1: Parallel Case Validation with Rayon

**Source:** Adapted from existing `BatchOracle::evaluate_population()` pattern (src/lib.rs:633-744)

```rust
// src/validation/ashrae_140_validator.rs
use rayon::prelude::*;

pub fn validate_all_parallel(
    &mut self,
    use_surrogates: bool,
    gpu_enabled: bool,
    gpu_devices: &[usize],
) -> BenchmarkReport {
    let cases = self.get_all_cases();
    let weather = Arc::new(DenverTmyWeather::new());

    // Optional: create shared GPU batcher if needed
    let shared_batcher = if use_surrogates && gpu_enabled {
        // Create once, share across all cases
        let surrogate = SurrogateManager::with_multi_device("models/surrogate.onnx", ...)
            .unwrap_or_else(|_| SurrogateManager::new().unwrap());
        let config = DynamicBatchConfig::high_throughput();
        Some(Arc::new(SharedBatchInferenceService::new(surrogate, config)))
    } else {
        None
    };

    let mut reports = Vec::new();

    rayon::scope(|s| {
        for case in cases {
            let weather = Arc::clone(&weather);
            let batcher = shared_batcher.as_ref().map(|b| Arc::clone(b));
            s.spawn(|_| {
                let report = self.validate_case_parallel(&case, &weather, use_surrogates, gpu_enabled, batcher);
                reports.push(report);
            });
        }
    });

    // Merge reports
    let mut final_report = BenchmarkReport::new();
    for report in reports {
        final_report.merge(report);  // TODO: implement merge()
    }
    final_report
}
```

### Example 2: GPU Autodetection and Configuration

**Source:** Based on `MultiDeviceSessionPool::detect_cuda_devices()` (src/ai/surrogate.rs:203-229)

```rust
// src/ai/surrogate.rs extension
impl SurrogateManager {
    pub fn auto_gpu(config: &ValidationConfig) -> Result<Self, String> {
        // Check feature flag
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!("CUDA feature not compiled; falling back to CPU");
            return Self::new();
        }

        // Check environment variable override
        if let Ok(env_override) = std::env::var("FLUXION_GPU") {
            if env_override == "0" {
                return Self::new();  // Force CPU
            }
        }

        // Check config file
        if !config.validation.gpu.enabled {
            return Self::new();  // Disabled in config
        }

        // Try to detect and create GPU session
        let devices = if config.validation.gpu.devices.is_empty() {
            MultiDeviceSessionPool::detect_cuda_devices().unwrap_or_else(|| vec![0])
        } else {
            config.validation.gpu.devices.clone()
        };

        if devices.is_empty() {
            eprintln!("No CUDA devices detected; falling back to CPU");
            return Self::new();
        }

        // Create multi-device pool (or single if only one device)
        let multi_config = MultiDeviceConfig {
            device_ids: devices,
            sessions_per_device: 4,
            auto_select: false,
            enable_affinity: true,
            fallback_to_cpu: true,
            max_retries: 3,
        };

        Self::with_multi_device("models/surrogate.onnx", multi_config)
            .or_else(|e| {
                eprintln!("GPU setup failed: {}; falling back to CPU", e);
                Self::new()
            })
    }
}
```

### Example 3: Performance History Recording

```rust
// src/validation/analyzer.rs
#[derive(Serialize, Deserialize)]
pub struct PerformanceRecord {
    timestamp: String,
    git_sha: String,
    mae: f64,
    max_deviation: f64,
    pass_rate: f64,
    validation_time_seconds: f64,
    throughput_configs_per_sec: f64,
    gpu_enabled: bool,
}

impl PerformanceRecord {
    pub fn append_to_history(&self) -> Result<(), String> {
        let path = "target/performance_history.jsonl";
        let json = serde_json::to_string(self).map_err(|e| e.to_string())?;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| e.to_string())?;
        use std::io::Write;
        writeln!(file, "{}", json).map_err(|e| e.to_string())?;
        Ok(())
    }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Sequential case validation (`for case in cases`) | Parallel case validation (`cases.par_iter()`) | Phase 6 | 5-10× speedup on 8-core |
| CPU-only surrogate inference | GPU-accelerated CUDA inference | Phase 6 | 10-100× speedup per inference batch |
| No batching within time-first loop | `SharedBatchInferenceService` aggregates across cases | Phase 6 | Better GPU occupancy, reduces small-batch penalty |
| No performance tracking | `BenchmarkReport` includes timing & throughput | Phase 6 | Enables regression detection |
| No guardrails | Threshold-based alerts in CI | Phase 6 | Prevents silent degradations |
| Manual baseline comparison | Automatic baseline + historical trend | Phase 6 | Systematic performance management |

**Deprecated/outdated:**
- Running `fluxion validate` without parallelism on modern hardware (unacceptably slow)
- Using mock surrogate (constant 1.2) for benchmarks (invalidates accuracy)
- Relying on `cargo bench` alone for validation performance (doesn't test end-to-end)

## Open Questions

1. **Should we use async (tokio) for SharedBatchInferenceService or pure threads?**
   - What we know: `crossbeam` channels + dedicated thread is simpler, sufficient for 18→1 case aggregation.
   - What's unclear: Could async handle more complex backpressure or integrate with async I/O? Not needed currently.
   - Recommendation: Use pure threads with crossbeam channels. Add tokio later if needed for async file I/O or web service.

2. **What is the exact target baseline for guardrails?**
   - What we know: Baseline needs to be established on reference hardware (8-core CPU, RTX 4090 or similar).
   - What's unclear: How much hardware variance to tolerate? Should baseline be per-CI-runner type?
   - Recommendation: Run baseline on clean checkout (no changes) 3×, take median. Store in `docs/performance_baseline.json`. Guardrails compare to stored baseline, not previous CI run.

3. **How to handle surrogate training data generation when surrogates are not yet trained?**
   - What we know: `SurrogateManager::new()` returns mock (1.2 constant) when no model loaded.
   - What's unclear: Should validation fail if surrogates enabled but no model present? Or fall back to analytical?
   - Recommendation: Graceful fallback: if model file missing, log warning and run analytical. Surrogate integration is opt-in via config; default should be analytical until models are trained and validated.

## Validation Architecture

> Nyquist validation is enabled per `.planning/config.json`.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in test (cargo test) + integration tests |
| Config file | `src/validation/` module tests |
| Quick run command | `cargo test --release test_parallel_validation` |
| Full suite command | `cargo test --release` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| BATCH-01 | Parallel case execution with rayon | integration | `cargo test --release test_parallel_validation --nocapture` | ❌ Wave 0 |
| GPU-01 | CUDA backend session creation | unit | `cargo test --release surrogate_gpu_setup --nocapture` | ❌ Wave 0 |
| SURR-01 | Session pool returns sessions correctly | unit | `cargo test --release test_session_pool` | ✅ src/ai/surrogate.rs:687-716 |
| SURR-02 | Batched inference produces correct outputs | integration | `cargo test --release test_predict_loads_batched` | ✅ src/ai/surrogate.rs:777-792 |
| REG-01 | MAE threshold guardrail fails on regression | unit | `cargo test --release test_guardrail_mae` | ❌ Wave 0 |
| REG-04 | Performance history appends to JSONL | unit | `cargo test --release test_performance_history` | ❌ Wave 0 |

**Missing test infrastructure (Wave 0 gaps):**
- [ ] `tests/test_parallel_validation.rs` — verify cases run in parallel (check thread count, duration vs sequential)
- [ ] `tests/test_shared_batch_service.rs` — verify request aggregation, batch formation, result routing
- [ ] `tests/test_guardrails.rs` — test threshold comparisons, CI exit codes
- [ ] `tests/test_performance_history.rs` — test JSONL append, file creation, format validation
- [ ] `tests/gpu/integration_test.rs` — conditionally skip if no GPU: test CUDA session creation, OOM handling (if we want to simulate)

**Note:** Existing surrogate tests (src/ai/surrogate.rs:687-792) cover basic SessionPool and batched inference. Those are unit tests with mock/real ONNX. Phase 6 tests need to cover *parallel coordination* and *guardrails*.

### Sampling Rate

- **Per task commit:** `cargo test --release test_parallel_validation` (should complete <30s)
- **Per wave merge:** Full test suite `cargo test --release` (may take few minutes)
- **Phase gate:** Run `fluxion validate --all --gpu` (if GPU available) and verify <5 minutes total; check `target/performance_history.jsonl` appended

### Wave 0 Gaps

- [ ] `tests/test_parallel_validation.rs` — cover BATCH-01
- [ ] `tests/test_shared_batch_service.rs` — cover SURR-02 refinement
- [ ] `tests/test_guardrails.rs` — cover REG-01, REG-02
- [ ] `tests/test_performance_history.rs` — cover REG-04
- [ ] `tests/gpu/setup_test.rs` — cover GPU-01 (skip if `!(feature = "cuda")`)
- [ ] Benchmark: `benches/validation_performance.rs` to establish baseline (optional)

*(If gaps filled: "Pre-built test scaffolds cover all phase requirements")*

## Integration Specifics

### Files to Modify

1. **`src/validation/ashrae_140_validator.rs`**
   - Add `validate_all_parallel()` method that uses `rayon::par_iter()` over cases (BATCH-01)
   - Add timing instrumentation: record `Instant::now()` before/after each case, store in report (BATCH-03, REG-03)
   - Accept `ValidationConfig` with `gpu_enabled`, `gpu_devices`, `use_surrogates` flags
   - Conditionally create `SharedBatchInferenceService` when `use_surrogates && gpu_enabled`

2. **`src/validation/report.rs`**
   - Extend `BenchmarkReport`:
     - Add `performance_metrics: HashMap<String, PerformanceMetrics>` field
     - Add `metadata: ReportMetadata` (timestamp, git_sha, config)
   - Add `record_performance()` method
   - Add `total_validation_time()` method
   - Update `to_markdown()` to include Performance Summary section

3. **`src/validation/reporter.rs`**
   - Update `render_markdown()` to output performance section after accuracy summary: table with case times, total time, throughput estimate
   - Add guardrail comparison: compare MAE/MaxDev to baseline, mark with ✅/⚠️/❌

4. **`src/lib.rs`** (BatchOracle)
   - Add `BatchOracle::set_gpu_enabled(flag: bool)` method (or accept in constructor)
   - Modify `evaluate_population()` to accept optional `SharedBatchInferenceService` parameter; if provided, use it for cross-case batching instead of internal coordinator-worker
   - Preserve existing coordinator-worker for single-Model usage

5. **`src/ai/surrogate.rs`**
   - Enhance `SurrogateManager::new()` to respect `FLUXION_GPU` env var and config (GPU-01)
   - Add `get_inference_metrics() -> InferenceMetrics` method to expose peak memory, throughput
   - Ensure `predict_loads_batched()` is thread-safe when called from multiple rayon workers (it is, via `SessionPool`)

6. **`src/ai/batch_inference.rs`**
   - Extend `DynamicBatchConfig` with `max_wait_ms` default 10ms (already exists) but ensure `SharedBatchInferenceService` respects it
   - Add `SharedBatchInferenceService` implementation (GPU-02, BATCH-01)

7. **`src/bin/fluxion.rs`**
   - Add CLI options:
     - `--gpu` / `--no-gpu` (override)
     - `--ci` (non-interactive output)
     - `--benchmark` (detailed microbenchmarks)
   - Pass config to validator: `ASHRAE140Validator::with_config(validation_config)`
   - After validation, print guardrail summary and exit with non-zero code if accuracy regression

8. **`src/validation/analyzer.rs`** (new or extend)
   - Create `GuardrailChecker` that loads baseline from `docs/performance_baseline.json`
   - Compare current `BenchmarkReport` to baseline, produce `GuardrailReport` with pass/fail/warn
   - Integrate with CLI to set exit code

9. **`docs/performance_baseline.json`** (new file)
   - Store baseline metrics: hardware description, mae, max_deviation, pass_rate, validation_time_seconds, throughput
   - Not committed to git (inconsistent hardware) but template committed

### New Files to Create

| File | Purpose | Implementation Notes |
|------|---------|---------------------|
| `src/ai/shared_batch_service.rs` | Cross-case GPU batching coordinator | Use `crossbeam::SegQueue`, dedicated thread, `DynamicBatchConfig`. Ensure proper shutdown via `Drop`. |
| `src/validation/config.rs` | Validation configuration structs | `ValidationConfig` with `gpu.enabled`, `gpu.devices`, `batch.*`, `guardrails.*`, `surrogate.*` fields. Load from environment + optional config file (`validation.yaml` in project root). |
| `tests/test_parallel_validation.rs` | Verify case-level parallelism speedup | Run sequential vs parallel on 18 cases, assert parallel faster by expected factor (e.g., >1.5× on 2 cores). |
| `tests/test_shared_batch_service.rs` | Verify request aggregation and backpressure | Submit many requests, check batch sizes, ensure no deadlock. |
| `tests/test_guardrails.rs` | Verify threshold logic | Test MAE >2% triggers fail, >10% MaxDev triggers fail, pass rate drop triggers fail. |
| `tests/test_performance_history.rs` | Verify JSONL append and format | Write temp file, append multiple records, parse back. |
| `benches/validation_performance.rs` | Microbenchmark to establish baseline | Benchmark `validate_all_parallel` with varying thread counts, batch sizes. |
| `tools/train_surrogate.py` | Training pipeline (already planned) | Generate dataset, train modular ONNX models. Not Phase 6 core, but needed for GPU-02 demonstration. |
| `docs/performance_baseline.json` | Baseline metrics for guardrails | Populate after first successful run. |

### Configuration Schema

Recommended `validation.yaml` (optional, defaults from environment):

```yaml
validation:
  gpu:
    enabled: auto    # auto|true|false
    devices: [0]     # list of GPU device IDs
  batch:
    max_batch_size: 512
    min_batch_size: 16
    max_wait_ms: 10   # force batch after 10ms even if not full
    target_batch_size: 128
    enable_adaptation: true
  guardrails:
    fail_on_accuracy_regression: true
    mae_threshold: 0.02      # 2%
    max_deviation_threshold: 0.10  # 10%
    pass_rate_drop_threshold: 0.05  # 5 percentage points
    time_warn_factor: 1.10   # 110%
  surrogate:
    enabled: false   # default off until fully validated
    sanity_check: true
    sanity_window: 100  # timesteps
    sanity_tolerance: 0.10  # 10%
    models:
      solar: "models/solar_gains_surrogate.onnx"
      hvac: "models/hvac_loads_surrogate.onnx"
      infiltration: "models/infiltration_surrogate.onnx"
```

Load via `src/validation/config.rs` using `serde_yaml`. Fallback to environment variables (`FLUXION_VALIDATION_GPU_ENABLED`, etc.) and hardcoded defaults.

## Performance Benchmarks

### Target Metrics

| Metric | Target (8-core + GPU) | Measurement Method |
|--------|----------------------|--------------------|
| Total validation time | <300 seconds (5 minutes) | `validate_all_parallel()` total `Instant::now()` duration |
| Throughput (cases/sec) | >3.6 cases/sec (18 cases / 5 min) | `18 / total_time_secs` |
| Surrogate inference latency (batch 512) | <10ms | `InferenceMetrics::avg_inference_time_ms` |
| GPU memory usage | <4GB peak | `nvidia-smi` during run or CUDA API (not critical) |
| CPU utilization | >80% average | `htop` (observational) |

### Baseline Establishment Process

1. **Hardware specification:** Document exact hardware: CPU model, core count, GPU model, RAM, CUDA version, driver version. Example: "Intel i9-13900K (24 cores), NVIDIA RTX 4090 24GB, CUDA 12.4".
2. **Warm-up runs:** Execute `fluxion validate --all` three times consecutively on clean checkout (no code changes) to let caches stabilize.
3. **Record metrics:**
   - Total wall-clock time (`time fluxion validate --all`)
   - MAE, MaxDev, PassRate from validation (accuracy should be stable)
   - GPU memory peak (check `nvidia-smi` logs)
   - Throughput (cases/sec)
4. **Store in `docs/performance_baseline.json`:**

```json
{
  "timestamp": "2026-03-10T12:00:00Z",
  "git_sha": "abc1234",
  "hardware": {
    "cpu": "Intel i9-13900K",
    "cpu_cores": 24,
    "gpu": "NVIDIA RTX 4090",
    "gpu_memory_gb": 24,
    "cuda_version": "12.4"
  },
  "mae": 10.5,
  "max_deviation": 15.2,
  "pass_rate": 100.0,
  "validation_time_seconds": 245.1,
  "throughput_configs_per_sec": 12345,
  "num_cases": 18,
  "surrogate_version": "v1.0.0"
}
```

5. **Guardrails:** Compare future runs to these absolute values. If hardware differs, adjust thresholds proportionally or skip performance guardrails.

### Regression Detection Logic

```rust
// src/validation/analyzer.rs
pub struct GuardrailChecker {
    baseline: PerformanceBaseline,
    config: GuardrailConfig,
}

impl GuardrailChecker {
    pub fn check(&self, report: &BenchmarkReport, perf: &PerformanceRecord) -> GuardrailReport {
        let mut results = GuardrailReport::new();

        // Accuracy regressions (fail CI)
        let mae_increase = (report.mae() - self.baseline.mae) / self.baseline.mae;
        if mae_increase > self.config.mae_threshold {
            results.add_failure(format!("MAE increased by {:.1}% (threshold >{:.1}%)",
                mae_increase * 100.0, self.config.mae_threshold * 100.0));
        }

        let maxdev_increase = (report.max_deviation() - self.baseline.max_deviation) / self.baseline.max_deviation;
        if maxdev_increase > self.config.max_deviation_threshold {
            results.add_failure(format!("Max Deviation increased by {:.1}% (threshold >{:.1}%)",
                maxdev_increase * 100.0, self.config.max_deviation_threshold * 100.0));
        }

        let passrate_drop = self.baseline.pass_rate - report.pass_rate();
        if passrate_drop > self.config.pass_rate_drop_threshold * 100.0 {
            results.add_failure(format!("Pass rate dropped by {:.1} percentage points (threshold >{:.1}pp)",
                passrate_drop, self.config.pass_rate_drop_threshold * 100.0));
        }

        // Performance regressions (warn only)
        let time_increase = (perf.validation_time_seconds - self.baseline.validation_time_seconds) / self.baseline.validation_time_seconds;
        if time_increase > self.config.time_warn_factor - 1.0 {
            results.add_warning(format!("Validation time increased by {:.1}% (baseline: {:.1}s, current: {:.1}s)",
                time_increase * 100.0, self.baseline.validation_time_seconds, perf.validation_time_seconds));
        }

        results
    }
}
```

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GPU driver/CUDA version mismatch causing runtime errors | Medium | High (validation fails) | Check CUDA availability at startup; fall back to CPU with warning; document CUDA requirements |
| Out-of-memory on GPU with batch size 512 | Medium | Medium | Implement OOM retry with reduced batch size; monitor memory; make `max_batch_size` configurable |
| Parallel validation slower than sequential due to cache thrashing | Low | Medium | Benchmark both; add `--no-parallel-cases` flag to disable case-level parallelism if needed |
| Race condition in `SharedBatchInferenceService` causing incorrect results | Medium | High | Extensive testing: property-based testing with `proptest`, deterministic results check |
| Guardrails produce false positives due to hardware variance | High | Medium | Tune thresholds after initial baseline runs; use relative performance metrics or hardware-normalized baselines |
| Surrogate accuracy unacceptable (>15% error on cases) | Medium | High | Fallback to analytical automatically; modular surrogates allow component-wise validation and isolation |
| Nested parallelism panics due to rayon scope misuse | Low | High | Use existing `BatchOracle` pattern as template; `rayon::scope()` for inner loops, no custom thread pool creation |

### Fallback Plans

- **GPU unavailable:** Auto-fallback to CPU surrogate inference (existing `SurrogateManager::new()` returns mock). No validation impact, just slower.
- **SharedBatchInferenceService deadlocks:** Fall back to per-case `SurrogateManager` instances (no cross-case batching). Accept lower GPU utilization but still correct.
- **Parallelism performance worse:** Add CLI flag `--sequential-cases` to disable case-level parallelism, keep population-level only.
- **Guardrails too strict:** Add `--no-guardrails` flag to bypass; tune thresholds in `validation.yaml`.

## Dependencies

### System Requirements

- **CUDA:** NVIDIA driver ≥ 525.x (CUDA 12.x), CUDA Toolkit 11.8+ installed (for GPU builds). The `ort` crate with `download-binaries` feature bundles CUDA libraries but requires system driver.
- **GPU Memory:** ≥8GB VRAM recommended for `max_batch_size=512`. 4GB may work with smaller batches.
- **CPU:** 8+ cores recommended for target <5min execution.
- **Rust:** 1.70+ (edition 2021)
- **Python:** Only for training surrogates (optional for Phase 6), maturin for PyO3 builds.

### Crates to Add

```toml
# Cargo.toml
[dependencies]
indicatif = { version = "0.17.7", optional = true }  # for progress bars
serde_yaml = { version = "0.9", optional = true }   # for validation.yaml config

[features]
default = []
progress-bars = ["indicatif"]
config-yaml = ["serde_yaml"]
```

These are optional; Phase 6 can proceed without them initially. Progress bars enhance UX but not required for CI. `serde_yaml` only if we want YAML config file.

## Incremental Rollout Plan

1. **Week 1: Parallel Case Execution (BATCH-01)**
   - Implement `validate_all_parallel()` with rayon
   - Add timing instrumentation, `PerformanceMetrics` struct
   - Test on CI with --release
   - **Success metric:** 2× speedup on 4-core, 4× on 8-core (without GPU yet)

2. **Week 2: SharedBatchInferenceService (GPU-02, SURR-02)**
   - Implement `src/ai/shared_batch_service.rs`
   - Integrate with `BatchOracle` and validation
   - Test with mock surrogates first (constant output)
   - **Success metric:** GPU utilization >80% during validation (visible in `nvidia-smi`)

3. **Week 3: GPU Autodetection and Configuration (GPU-01, SURR-01)**
   - Extend `SurrogateManager::new()` with env/config flags
   - Add CLI flags `--gpu`, `--no-gpu`
   - Test on GPU machine, verify fallback behavior
   - **Success metric:** `fluxion validate --all --gpu` runs successfully on GPU without code changes

4. **Week 4: Guardrails and Historical Tracking (REG-01 through REG-04)**
   - Implement `GuardrailChecker`, load baseline from JSON
   - Add `performance_history.jsonl` append after each validation
   - Integrate with CLI exit codes: accuracy failure exits 1, performance warning exits 0
   - **Success metric:** CI detects intentional MAE degradation (>2%) and fails build

5. **Week 5: Training Pipeline and Model Validation (SURR-03)**
   - Create `tools/train_surrogate.py` (outside Rust scope)
   - Generate dataset from analytical runs, train modular surrogates
   - Validate each surrogate on holdout set (<5% error)
   - Test surrogate-enabled validation: `fluxion validate --all --use-surrogates --gpu`
   - **Success metric:** Surrogate accuracy within 15% per case; speedup >10× vs analytical

6. **Week 6: Benchmarks and Baseline Establishment**
   - Run on reference hardware, populate `docs/performance_baseline.json`
   - Add `cargo bench` benchmarks for micro-benchmarks
   - Final tuning: adjust batch sizes, thread counts based on profiling
   - **Success metric:** Total validation time <5 minutes on reference hardware with GPU surrogates

### Safe Defaults

- **Default mode:** No GPU, no surrogates, sequential case execution. This ensures validation always works on minimal hardware.
- **`--gpu` flag:** Enables GPU for surrogates *if* model is present and CUDA available. Otherwise falls back to CPU with warning.
- **`--use-surrogates` flag:** Opt-in. Default false (analytical). Surrogates must be pre-trained and placed in `models/` directory.
- **Parallelism:** Auto-enabled when `RAYON_NUM_THREADS` > 1 (which is default). Can be disabled with `RAYON_NUM_THREADS=1`.
- **Guardrails:** Enabled by default but use `warn` for performance, `fail` only for accuracy. Thresholds conservative (2% MAE, 10% MaxDev) to avoid false positives early.

## Sources

### Primary (HIGH confidence)
- `src/ai/surrogate.rs` - `SessionPool`, `MultiDeviceSessionPool`, CUDA backend integration
- `src/ai/batch_inference.rs` - `DynamicBatchManager`, `DynamicBatchConfig` for adaptive batching
- `src/lib.rs` - `BatchOracle::evaluate_population()` coordinator-worker pattern, `rayon::scope` usage
- `src/validation/ashrae_140_validator.rs` - `validate_analytical_engine()` sequential baseline to parallelize
- `src/validation/report.rs` - `BenchmarkReport` structure, MAE/max_deviation calculations

### Secondary (MEDIUM confidence)
- `CLAUDE.md` (project instructions) - Batch Oracle pattern, parameter semantics, profiling guidelines
- `.planning/phases/06-performance-optimization/06-CONTEXT.md` - Locked decisions for Phase 6 scope and implementation approach
- `src/validation/benchmark.rs` - Reference data structure and case specifications

### Tertiary (LOW confidence)
- Web search results attempted for ONNX Runtime CUDA optimization and rayon nested parallelism (unavailable due to model restrictions)
- General knowledge of ONNX Runtime session pooling and GPU batching best practices from documentation (not directly verified)

## Metadata

**Confidence breakdown:**
- Standard stack (rayon, ort, crossbeam): HIGH - Existing dependencies, proven in codebase
- Architecture (hybrid parallel, shared batcher): HIGH - Based on existing BatchOracle pattern and SessionPool, integration is logical extension
- Integration specifics: HIGH - File locations and method signatures identified from codebase
- Performance benchmarks: MEDIUM - Targets are estimates; need baseline runs to confirm
- Risk analysis: HIGH - Pitfalls identified from existing parallel code and ONNX patterns
- Test strategy: HIGH - Leverages existing test patterns; new tests needed for parallelism and guardrails

**Research date:** 2026-03-10
**Valid until:** 2026-04-10 (30 days; stable dependencies)

## RESEARCH COMPLETE
