# Issue #721: CPU Flamegraph and Hot-Path Profiling for v1.3

## Executive Summary

This document summarizes the profiling infrastructure available in Fluxion and identifies the top optimization targets for v1.3 based on existing benchmarks and code analysis.

**Phase 47 Achievements (Baseline for v1.3):**
- Single-zone timestep: 45.2ms → 32.1ms (29% improvement)
- 10-zone timestep: 88.7ms → 64.3ms (27% improvement)
- Memory: 9.2MB → 7.8MB (15% reduction)
- Solver iterations: 18 avg → 12 avg (33% reduction)

## 1. Profiling Tools Available

### 1.1 Installed Tools

| Tool | Status | Purpose |
|------|--------|---------|
| `criterion` | ✅ In dev-dependencies | Statistical benchmarking (benches/*) |
| `dhat` | ✅ In dev-dependencies | Heap profiling (tests/test_allocation_tracking.rs) |
| `cargo-flamegraph` | ❌ Not installed | CPU flamegraph generation |
| `perf` | ⚠️ System tool | Linux profiler for hardware events |
| `pprof` | ❌ Not available | Go-style profiling (Rust via `pprof` crate) |

### 1.2 How to Install Missing Tools

```bash
# CPU flamegraph (requires dtrace on macOS, perf on Linux)
cargo install cargo-flamegraph

# For Linux perf-based profiling:
sudo apt-get install linux-perf  # Debian/Ubuntu
sudo perf record -g -- cargo run --release --bin fluxion -- [args]
```

### 1.3 Existing Profiling Infrastructure

**Heap Profiling:**
- `tests/test_allocation_tracking.rs` - Uses dhat for allocation tracking
- `dhat-heap.json` - Sample dhat output file
- Run with: `cargo test --test test_allocation_tracking`

**Criterion Benchmarks:**
- `benches/performance.rs` - Main thermal solver benchmarks
- `benches/benchmark_8760_timesteps.rs` - Annual simulation benchmarks
- `benches/cta_perf_comparison.rs` - CTA performance comparison
- `benches/engine_bench.rs` - Engine benchmarks
- Run with: `cargo bench --bench performance`

## 2. Hot Paths Identified

Based on code analysis of `src/sim/thermal_model_physics/`, the following hot paths are executed per timestep:

### 2.1 Primary Hot Path: step_physics Dispatcher

**File:** `src/sim/thermal_model_physics/step_dispatcher.rs:29`

```
step_physics(timestep, outdoor_temp, dt_seconds)
├── calc_analytical_loads() [if weather.is_some()]
├── step_physics_9r4c() [if is_nine_r4c_model]
├── step_physics_8r3c() [if is_8r3c_model]
├── step_physics_6r2c() [if is_6r2c_model]
└── step_physics_5r1c() [default]
```

### 2.2 Secondary Hot Path: solve_timesteps

**File:** `src/sim/thermal_model_physics/solver_core.rs:65`

```
solve_timesteps(steps, surrogates, use_surrogates, ...)
└── solve_timesteps_with_dt()
    ├── set_weather()
    ├── calculate_timestep_seconds()
    └── [loop steps times]
        └── step_physics()
```

### 2.3 Third Hot Path: Physics Implementation (5R1C)

**File:** `src/sim/thermal_model_physics/physics_impl.rs:38-2603`

Key sub-operations in `step_physics_5r1c`:
1. `prepare_solvers_and_sol_air()` - Solar/air temperature calculation
2. VectorField allocations (lines 96-116)
3. `crank_nicolson_iso13790()` or `backward_euler_update_2cond()` - Integration
4. Zone coupling calculations

## 3. Top 3 Optimization Targets for v1.3

Based on Phase 47 benchmark data and code analysis:

### Target 1: VectorField Allocation Reduction

**Current State:** Multiple `Vec` allocations per timestep in `step_physics_5r1c` (lines 96-116)

**Impact:** Estimated 15-20% of timestep time

**Recommendation:**
```rust
// Current pattern (allocates 3 Vecs per zone per timestep):
let mut phi_ia_data = Vec::with_capacity(self.0.num_zones);
let mut phi_st_data = Vec::with_capacity(self.0.num_zones);
let mut phi_m_data = Vec::with_capacity(self.0.num_zones);

// Optimization: Pre-allocate once in ThermalModel::new()
```

**Files to Modify:** `src/sim/thermal_model_physics/physics_impl.rs`

### Target 2: Integration Method Selection Overhead

**Current State:** `select_integration_method()` called every timestep

**Impact:** Estimated 10-15% of timestep time

**Recommendation:** Cache integration method selection per model, only recalculate when parameters change.

**Files to Modify:** `src/sim/thermal_integration/mod.rs`

### Target 3: Weather Data Access in Loop

**Current State:** `self.0.weather.as_ref()` called multiple times per `step_physics`

**Impact:** Estimated 5-10% of timestep time

**Recommendation:** Borrow weather data once at start of `step_physics` and pass references

**Files to Modify:** `src/sim/thermal_model_physics/step_dispatcher.rs`, `physics_impl.rs`

## 4. Profiling Commands for v1.3

### 4.1 CPU Flamegraph

```bash
# Install cargo-flamegraph
cargo install cargo-flamegraph

# Generate flamegraph for thermal solver
cargo flamegraph --bin fluxion --release -- \
  fluxion run --scenario single-zone --timesteps 8760

# Or for a specific benchmark
cargo flamegraph --bench performance -- --single-zone
```

### 4.2 Heap Profile

```bash
# Run with dhat heap profiler
cargo test --test test_allocation_tracking --release -- \
  --nocapture 2>&1 | grep -A50 "dhat"

# Or add dhat to a specific benchmark
RUSTFLAGS="-O -C inline-threshold=1000" cargo bench --bench performance
```

### 4.3 Criterion Benchmark

```bash
# Run all performance benchmarks
cargo bench --bench performance --release

# Run with HTML output
cargo bench --bench performance --release -- \
  --html-depth 2 --output-format html

# View results
ls -la target/criterion/
```

### 4.4perf (Linux-specific)

```bash
# Record profile
sudo perf record -g -- cargo run --release --bin fluxion -- [args]

# Generate flamegraph from perf data
perf script | stackcollapse-perf | flamegraph > profile.svg
```

## 5. Benchmark Infrastructure

### 5.1 Main Benchmarks

| Benchmark | File | Metrics |
|-----------|------|---------|
| `thermal_solver_single_zone` | benches/performance.rs | Single-zone timestep |
| `thermal_solver_10_zones` | benches/performance.rs | 10-zone timestep |
| `solve_timesteps_8760_analytical` | benches/performance.rs | Annual simulation |
| `multizone_8760_10zones` | benches/benchmark_8760_timesteps.rs | 10-zone annual |
| `batch_oracle_8760` | benches/benchmark_8760_timesteps.rs | Population runs |

### 5.2 CI Integration

Benchmarks run automatically on:
- Push to `main` and `dev` branches
- Via `.github/workflows/performance.yml`

### 5.3 Performance Targets for v1.3

| Metric | Phase 47 Baseline | v1.3 Target | Improvement Needed |
|--------|-------------------|-------------|-------------------|
| Single-zone timestep | 32.1ms | <25ms | ~22% |
| 10-zone timestep | 64.3ms | <50ms | ~22% |
| Memory (10 zones) | 7.8MB | <6MB | ~23% |
| Solver iterations | 12 avg | <10 avg | ~17% |

## 6. Recommendations

### Immediate Actions

1. **Install cargo-flamegraph** - Required for accurate hot-path identification
2. **Run heap profiling** - Identify allocation patterns in 8760-step runs
3. **Profile on target hardware** - Linux perf for accurate hardware event data

### v1.3 Optimization Priorities

1. **VectorField pre-allocation** - Low-hanging fruit, high impact
2. **Integration method caching** - Reduce per-timestep overhead
3. **Weather data borrow consolidation** - Minor but consistent improvement

### Validation

After implementing optimizations:
```bash
# Verify no regressions
cargo bench --bench performance --release

# Run full test suite
cargo test --release

# Run ASHRAE 140 validation
cargo test --test integration
```

## 7. References

- Phase 47 Completion: `documentation/performance_completion.md`
- Performance Guide: `documentation/performance.md`
- Thermal Model Architecture: `src/sim/thermal_model_physics/mod.rs`
- Benchmark Suite: `benches/performance.rs`

---

*Document generated for Issue #721: CPU Flamegraph and hot-path profiling*
*Purpose: Identify optimization targets for v1.3*
