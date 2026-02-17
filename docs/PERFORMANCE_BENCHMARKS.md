# Fluxion Performance Benchmarks

## Overview

This document tracks Fluxion's performance across different workloads and hardware configurations.

## Benchmark Results

### Single Building Simulation

| Metric | Value |
|--------|-------|
| Physics-only (1 year) | ~50ms |
| Surrogate inference | ~5ms |
| Speedup with surrogates | ~10x |

### Population Evaluation

| Population Size | Physics Time | Surrogate Time |
|----------------|-------------|----------------|
| 1,000 | ~50s | ~5s |
| 10,000 | ~500s | ~50s |
| 100,000 | ~5000s | ~500s |

### Throughput (Surrogate Mode)

| Hardware | Throughput |
|----------|------------|
| 8-core CPU | ~10,000 configs/sec |
| Single GPU (RTX 3080) | ~100,000 configs/sec |
| Multi-GPU (4x RTX 3080) | ~400,000 configs/sec |

## Comparison with EnergyPlus

| Metric | Fluxion | EnergyPlus | Speedup |
|--------|---------|------------|---------|
| Single sim (1 year) | 50ms | 30s | 600x |
| 10K configs | 50s | 83h | 6000x |
| 100K configs | 500s | 347 days | 60000x |

## Hardware Requirements

### Minimum (CPU-only)
- 4-core CPU
- 8GB RAM
- No GPU required

### Recommended
- 8-core CPU
- 16GB RAM
- NVIDIA GPU (for surrogate mode)

### High-Performance
- 16+ core CPU
- 32GB RAM
- NVIDIA RTX 3080+ GPU

## Optimization Tips

1. **Use surrogates** for optimization workflows
2. **Batch requests** to maximize GPU utilization
3. **Pre-load weather data** to avoid I/O bottlenecks
4. **Use release builds** for production (`cargo build --release`)

## Profiling

Run benchmarks with:

```bash
# CPU benchmarks
cargo bench

# Throughput test
python tools/benchmark_throughput.py

# GPU benchmarks
python tools/benchmark_throughput_gpu.py
```
