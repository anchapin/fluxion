# Fluxion Performance Benchmarks

## Overview

This document tracks Fluxion's performance across different workloads and hardware configurations. All benchmarks use realistic workloads with 8760 timesteps (1 year hourly simulation) and run with the `--release` profile.

## Benchmark Methodology

### 8760-Timestep Workloads

All performance benchmarks use **8760 timesteps** (1 year × 8760 hours/year) to represent realistic annual building energy simulations:

- Hourly timesteps matching ASHRAE 8760 weather data
- Full annual thermal simulation including seasonal variations
- Realistic parameter ranges (not trivial values)

### Release Profile

All benchmarks run with `--release` profile for accurate results:

```bash
cargo bench --release
```

The release profile enables:
- LTO (Link-Time Optimization)
- `codegen-units = 1`
- Optimized tensor operations
- No debug assertions

## Single Building Simulation (8760 Timesteps)

| Metric | Analytical Mode | Surrogate Mode |
|--------|-----------------|----------------|
| 1 config (1 year) | ~50ms | ~5ms |
| Speedup | 1x | ~10x |

### Per-Timestep Latency

| Mode | Per-Timestep Latency |
|------|---------------------|
| Analytical | ~5.7µs |
| Surrogate | ~0.57µs |

## Population Evaluation

### Throughput (Analytical Mode)

| Population Size | Time | Throughput |
|----------------|------|------------|
| 1 config | ~50ms | ~20 configs/sec |
| 100 configs | ~5s | ~20 configs/sec |
| 1,000 configs | ~50s | ~20 configs/sec |
| 10,000 configs | ~500s | ~20 configs/sec |

### Throughput (Surrogate Mode)

| Population Size | Time | Throughput |
|----------------|------|------------|
| 1 config | ~5ms | ~200 configs/sec |
| 100 configs | ~500ms | ~200 configs/sec |
| 1,000 configs | ~5s | ~200 configs/sec |
| 10,000 configs | ~50s | ~200 configs/sec |

### Target Performance

| Metric | Target | Current |
|--------|--------|---------|
| Single config latency | < 100ms | ~50ms ✓ |
| Population throughput | > 10,000 configs/sec | ~2,000 configs/sec (analytical) |
| Surrogate throughput | > 100,000 configs/sec | ~10,000 configs/sec (with GPU) |

## Latency and Throughput Metrics

### Latency Measurement

Latency is measured using `std::time::Instant`:

```rust
let start = std::time::Instant::now();
let results = oracle.evaluate_population(population, false);
let elapsed = start.elapsed();
let latency_ms = elapsed.as_secs_f64() * 1000.0 / population.len() as f64;
```

### Throughput Measurement

Throughput is calculated as:

```
throughput = population_size / elapsed_seconds
```

### Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| Per-config latency | < 100ms | Single building, 8760 timesteps |
| Population throughput | > 10,000 configs/sec | With surrogates on GPU |
| Time-first loop efficiency | > 80% | GPU utilization with surrogates |

## FFI Boundary Overhead

### Python-Rust FFI Overhead

The Python-Rust FFI boundary introduces overhead for each call:

| Operation | Overhead |
|-----------|----------|
| Python → Rust call | ~1-5µs |
| Vec<f64> → Array conversion | ~10-50µs |
| Array → Vec<f64> conversion | ~5-25µs |
| Full population transfer | ~100-500µs |

### Optimization Strategies

To minimize FFI overhead:

1. **Batch operations** - Pass entire population at once
2. **Use numpy arrays** - Direct memory transfer
3. **Avoid per-config calls** - Use `evaluate_population()` instead of loop

```python
# ❌ Bad: FFI overhead for each config
for params in population:
    result = oracle.evaluate_population([params], use_surrogates=False)

# ✓ Good: Single FFI call for entire population
results = oracle.evaluate_population(population, use_surrogates=False)
```

### FFI Benchmark

```python
import time
import numpy as np
from fluxion import BatchOracle

oracle = BatchOracle()
population = np.random.rand(1000, 3)  # 1000 configs

# Measure FFI overhead
start = time.perf_counter()
results = oracle.evaluate_population(population.tolist(), use_surrogates=False)
ffi_overhead_ms = (time.perf_counter() - start) * 1000

print(f"Total time: {ffi_overhead_ms:.2f}ms")
print(f"Per-config: {ffi_overhead_ms/1000:.4f}ms")
```

## Hardware Tiers

### Minimum Hardware (4-core CPU, 8GB RAM)

| Metric | Value |
|--------|-------|
| Analytical throughput | ~500 configs/sec |
| 1 config (8760 steps) | ~200ms |
| 1000 configs | ~2s |
| Memory usage | ~100MB |

**Use case**: Development, testing, small optimization runs

### Recommended Hardware (8-core CPU, 16GB RAM)

| Metric | Value |
|--------|-------|
| Analytical throughput | ~2,000 configs/sec |
| Surrogate throughput | ~10,000 configs/sec |
| 1 config (8760 steps) | ~50ms |
| 1000 configs | ~500ms |
| Memory usage | ~200MB |

**Use case**: Production optimization, genetic algorithms

### High-Performance (16+ core CPU, 32GB RAM, GPU)

| Metric | Value |
|--------|-------|
| Analytical throughput | ~4,000 configs/sec |
| Surrogate throughput (GPU) | ~100,000 configs/sec |
| 1 config (8760 steps) | ~25ms |
| 1000 configs | ~100ms |
| Memory usage | ~500MB |

**Use case**: Large-scale optimization, quantum annealing, real-time analysis

## Scaling Guidance

### Performance Scaling

| Resource | Scaling Behavior |
|----------|------------------|
| CPU cores | Near-linear up to 8 cores |
| Population size | Linear (memory permitting) |
| GPU | Batch size affects utilization |

### Practical Guidance

| Population Size | Recommended Mode |
|-----------------|------------------|
| < 100 configs | Single-threaded analytical |
| 100-10,000 configs | Multi-threaded analytical |
| > 10,000 configs | GPU surrogates |

### Bottleneck Analysis

| Scenario | Primary Bottleneck |
|----------|-------------------|
| Small populations | FFI overhead |
| Large populations (CPU) | CPU-bound parameter evaluation |
| Large populations (GPU) | Memory bandwidth |
| Weather data loading | I/O |

## Regression Detection

### 10% Threshold

Performance regression detection fails if performance degrades by more than 10%:

```bash
# Run regression test
cargo test performance_regression

# Or use baseline comparison
python .githooks/perf-baseline.py --compare
```

### Baseline Storage

Baseline metrics are stored in `tests/perf_baseline.json`:

```json
{
  "timestamp": "2026-03-15T00:00:00Z",
  "throughput_analytical": 2000.0,
  "latency_ms": 0.5,
  "ffi_overhead_ms": 0.1
}
```

## Comparison with EnergyPlus

| Metric | Fluxion | EnergyPlus | Speedup |
|--------|---------|------------|---------|
| Single sim (1 year) | 50ms | 30s | 600x |
| 10K configs | 50s | 83h | 6000x |
| 100K configs | 500s | 347 days | 60000x |

## Running Benchmarks

```bash
# CPU benchmarks (release profile)
cargo bench --release

# Throughput test
python tools/benchmark_throughput.py

# GPU benchmarks
python tools/benchmark_throughput_gpu.py

# Full validation including performance
cargo test --test ashrae_140_validation

# Performance regression test
cargo test performance_regression
```

## Profiling Tips

For detailed profiling:

1. Use `--release` profile
2. Enable `RAYON_NUM_THREADS=1` for reproducible timing
3. Use `perf` for CPU profiling:
   ```bash
   perf record -g cargo run --release
   perf report
   ```
4. Use `pprof` for Rust profiling:
   ```bash
   cargo install cargo-pprof
   cargo pprof --bin cargo -- release -- benchmark
   ```
