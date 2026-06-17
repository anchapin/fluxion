# Surrogate vs Physics Benchmarking Results

**Issue**: #720 - Perf: Surrogate vs. physics benchmarking
**Date**: 2026-06-16
**Benchmark Suite**: `benches/surrogate_vs_physics_bench.rs`

## Benchmark Configuration

- **Hardware**: Standard x86_64 development machine
- **Rust**: Release mode (`--release`)
- **ONNX Model**: `assets/dummy_surrogate.onnx` (pass-through model)
- **Criterion Sample Size**: 1000 samples (timing), 100 samples (batched), 10 samples (8760-step)

## Raw Benchmark Results

### 1. ONNX Surrogate Inference Timing

| Benchmark | Time (µs) | Throughput |
|-----------|-----------|------------|
| `surrogate_onnx/single_inference_6zones` | 3.33 - 3.57 | ~1.74 M elem/s |
| `surrogate_timing/onnx_inference_6input` | 2.59 - 2.62 | ~2.2 M elem/s |

### 2. Batched ONNX Inference

| Batch Size | Time (µs) | Throughput |
|------------|-----------|------------|
| 1 | 3.59 - 4.30 | ~1.5 M elem/s |
| 10 | 5.79 - 7.37 | ~9.1 M elem/s |
| 100 | 7.51 - 7.63 | ~79 M elem/s |

**Key Finding**: Batch inference amortizes overhead - 100x batch is only ~2x slower than single inference.

### 3. Physics Step Timing (per zone)

| Configuration | Time (ns) | Throughput |
|--------------|-----------|------------|
| 1 zone | 328 - 332 | ~3.0 M elem/s |
| 5 zones | 505 - 510 | ~9.8 M elem/s |
| 10 zones | 649 - 657 | ~15.3 M elem/s |

### 4. Analytical Loads (fallback path)

| Benchmark | Time (ns) | Throughput |
|-----------|-----------|------------|
| `analytical_loads_6zones` | 57.7 - 58.2 | ~99 M elem/s |

### 5. 8760-Timestep Full Simulation (10 zones)

| Mode | Time (ms) |
|------|-----------|
| Physics analytical | 6.56 - 6.63 |
| Physics with surrogate | 6.64 - 6.81 |

## Analysis

### Key Finding: ONNX Overhead vs Physics

The benchmark reveals important insights about the current surrogate implementation:

1. **ONNX inference overhead** (~3 µs per call) is approximately **10x higher** than a single physics `step_physics` call (~330 ns per zone).

2. **Analytical fallback** (~58 ns for 6 zones) is **~50x faster** than ONNX inference.

3. **8760-step simulation**: When no real ONNX model is loaded (mock mode), the surrogate infrastructure adds negligible overhead because the mock fallback (`vec![1.2; temps.len()]`) is extremely fast.

### On the 10-100x Speedup Claim

The README in `examples/packs/surrogate_benchmarking/` claims a "10-100x speedup vs physics simulation." This benchmark verifies:

1. The **surrogate infrastructure** is in place and integrated into the Criterion bench suite.

2. **Without a real ONNX model** (current benchmark conditions), the surrogate adds minimal overhead because the mock fallback is essentially a no-op.

3. **With a real ONNX model**, the speedup would depend on:
   - The complexity of the physics being replaced (CFD/ray-tracing can be extremely expensive)
   - The ONNX model complexity and input size
   - Whether batched inference is used

### Benchmark Coverage

This benchmark suite covers:
- [x] ONNX surrogate inference timing (single and batched)
- [x] Physics step timing (single and multi-zone)
- [x] Analytical fallback timing
- [x] 8760-timestep full simulation comparison
- [ ] Real ONNX model vs CFD comparison (requires production surrogate model)

## Recommendations

1. **For accurate speedup verification**: A real production surrogate model (not the dummy pass-through) should be benchmarked against the actual physics it replaces.

2. **For production use**: The batch inference results show significant throughput improvements (79 M elem/s for batch=100), suggesting batched inference should be used when possible.

3. **CI Integration**: This benchmark is now wired into `cargo bench` and can be run in CI. The key metric to track is `surrogate_timing/onnx_inference_6input` which should remain below 10 µs.

## Files

- `benches/surrogate_vs_physics_bench.rs` - Criterion benchmark suite
- `examples/packs/surrogate_benchmarking/` - Pack with configuration and scripts
- `src/ai/surrogate.rs` - SurrogateManager implementation

## Running the Benchmarks

```bash
# Run all surrogate vs physics benchmarks
cargo bench --bench surrogate_vs_physics

# Run specific benchmark group
cargo bench --bench surrogate_vs_physics --group surrogate_onnx

# Run with profiling
cargo bench --bench surrogate_vs_physics --profile-time=10
```
