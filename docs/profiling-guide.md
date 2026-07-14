# Profiling Guide

> **TL;DR**: Performance profiling tools and techniques for Fluxion physics modules.
> **Key decisions**: Use cargo-flamegraph for flamegraphs, perf for hardware counters, memory_profiler for heap analysis | Guidance for AI agents doing performance work.
> **Owned by**: Performance team
> **Reviewed**: 2026-07-13

## Available Tools

| Tool | Purpose | Installation |
|------|---------|--------------|
| `cargo-flamegraph` | CPU flamegraphs | `cargo install flamegraph` |
| `perf` | Hardware performance counters | Linux `linux-tools` package |
| `memory_profiler` | Heap allocation tracking | `cargo install memory_profiler` |
| `tools/benchmark_throughput.py` | Throughput benchmarking | `pip install pandas matplotlib` |

## Profiling Specific Modules

### Weather Module (`src/sim/weather.rs`)
```bash
cargo flamegraph --bin fluxion -- --mode weather > weather_profile.svg
perf stat -e cycles,instructions,cache-misses -- ./target/release/fluxion --mode weather
```

### Solar Module (`src/sim/solar.rs`)
```bash
cargo flamegraph --bin fluxion -- --mode solar > solar_profile.svg
memory_profiler ./target/release/fluxion --mode solar
```

### Conduction Solver (`src/physics/solver_trait.rs`)
```bash
perf record -g -- ./target/release/fluxion --mode conduction
perf report --stdio
```

### Ventilation (`src/sim/ventilation.rs`)
```bash
cargo flamegraph --bin fluxion -- --mode ventilation > ventilation_profile.svg
```

### Zone Balance (`src/sim/thermal_model.rs`)
```bash
perf stat -e branches,branch-misses -- ./target/release/fluxion --mode zone
```

## Throughput Benchmarking

```bash
python tools/benchmark_throughput.py --module solar --iterations 1000 --output benchmark_results.csv
```

## Interpreting Results

- **Flamegraph peaks**: Functions consuming most CPU time — optimize these first
- **perf cache-misses**: Memory access patterns — consider caching
- **memory_profiler**: Heap allocations — reduce allocations in hot paths
- **Throughput**: Operations/second — aim for >10% improvement per iteration

## Guidance for AI Agents

When profiling code:
1. Always profile before and after changes
2. Run benchmarks 3+ times and use median values
3. Profile under realistic load conditions
4. Focus on the hottest module first (usually thermal_model or solar)
5. Report both absolute (ms) and relative (%) improvements
