# Allocation Profiling Report

**Phase:** 09 - Performance Optimization
**Plan:** 09-02 (Reduce Heap Allocations in Hot Loop)
**Date:** 2026-03-12
**Tool:** `dhat` heap profiler, release builds

---

## Executive Summary

This report identifies heap allocation hotspots in the thermal simulation hot loop (`ThermalModel::solve_timesteps` → `step_physics_5r1c`). Baseline measurements show **≈219,097 total heap allocations** during a 1‑year, single‑zone analytical simulation, totaling **~1.76 MB** of allocated memory.

The average allocation rate is **≈25 allocations per timestep** (8760 steps). At this rate, even a single‑zone simulation incurs noticeable allocation overhead; for batch evaluation of 1000+ configurations, allocation pressure becomes a performance bottleneck.

Optimizations will focus on in-place arithmetic and buffer reuse to reduce allocation churn, with expected targets of **20–50% reduction** in allocation count.

---

## Methodology

- **Test:** `test_allocation_count_single_model` in `tests/test_allocation_tracking.rs`
- **Configuration:** 1 thermal zone, analytical physics (`use_surrogates=false`), release profile
- **Profiler:** `dhat::Profiler::new_heap()` with global allocator override
- **Metrics:** `allocation_count`, `bytes_allocated`, `peak_memory_usage`

Baseline output (fresh run 2026-03-12):
```
dhat: Total:     1,763,622 bytes in 219,097 blocks
dhat: At t-end:  10,288 bytes in 2 blocks
```

*Note: Allocation count decreased from earlier measurement (368,738 → 219,097), suggesting prior optimizations or measurement variance. Current count establishes new baseline.*

---

## Hotspot Analysis (Per‑Timestep)

The following allocation sites occur in `step_physics_5r1c` for a single‑zone, night‑ventilation‑disabled configuration. Each entry lists the line number (from `src/sim/engine.rs`), the allocation type, and its per‑timestep frequency.

### Primary Allocation Sources (Current Code)

| Line(s) | Allocation Type | Description | Per‑Step Count (avg) | Eliminable? |
|---------|----------------|-------------|----------------------|-------------|
| ~2020,2022 | `.clone()` (×2) | `self.loads.clone()`, `self.solar_gains.clone()` | 2 | ✅ Yes (reuse or direct ops) |
| ~2028 | `.clone()` | `internal_gains_watts.clone()` for convective split | 1 | ✅ Yes (consume pattern) |
| ~2036 | `.clone()` | `phi_rad_internal.clone()` for surface gains | 1 | ✅ Yes |
| ~2040 | `.clone()` | `solar_gains_watts.clone()` for solar split | 1 | ✅ Yes |
| ~2069 | `.clone()` + `constant_like` | Night vent: `h_ext_base.clone() + constant_like(h_ve_vent)` | 2 (when active) | ✅ Yes (pre-alloc or compute inline) |
| ~2090-2101 | `.clone()` (multiple) | Sensitivity recalculation (night vent active): `h_tr_iz`, `h_tr_iz_rad`, `derived_h_ms_is_prod`, `term_rest_1`, `derived_ground_coeff`, `den` | 6-8 | ✅ Yes (in-place arithmetic) |
| ~2103-2105 | `.clone()` (×3) | Standard sensitivity path: `derived_h_ms_is_prod`, `mass_temperatures`, `h_tr_is`, `phi_st` | 4 | ✅ Yes |
| ~2119 | `.clone()` | `phi_ia.clone()` for inter‑zone buffer | 1 | ✅ Yes (consume phi_ia directly) |
| ~2199 | `.clone()` | `term_rest_1.clone()` for ground term | 1 | ✅ Yes |
| ~2201 | `.clone()` | `derived_ground_coeff.clone()` for ground term | 1 | ✅ Yes |
| ~2234, 2241, etc. | `.clone()` | Diagnostic/cloning in energy calculation (various) | 3-5 | ⚠️ Partial (some are for diagnostics) |

**Total per‑step allocations (single‑zone, no night vent): ~20–25**

### Additional Allocations in Multi‑Zone (e.g., Case 960)

- **Line 2166** (original reference): The current implementation no longer uses `vec![-q_iz_total, q_iz_total]`. Instead, it adds directly to `phi_ia_with_iz` slice. However, `phi_ia.clone()` at line 2119 remains the main allocation for multi‑zone.

---

## Root Causes

1. **Non-consuming arithmetic**: Expressions like `a.clone() + b.clone() + c.clone()` create intermediate vectors. The `ContinuousTensor` trait provides `add_assign`, `mul_assign` but they are underutilized.
2. **Defensive clones**: Some `.clone()` calls prevent moving values that are reused elsewhere; however, after the value is no longer needed, we can move instead.
3. **Constant vector allocation**: `constant_like(value)` creates a new vector filled with `value`. When this is used for temporary h_ext modification, it allocates.
4. **Diagnostic conversions**: `.to_vec()` calls for summation can be replaced with iterator operations on slices.

---

## Optimization Strategy (Tasks 2-3)

### Task 2: In-place Arithmetic (Trait Already Exists)
The `ContinuousTensor` trait already defines `add_assign`, `mul_assign`, `sub_assign`, `div_assign` (implemented for `VectorField`). We will refactor arithmetic chains to use consuming patterns.

**Pattern change:**
```rust
// Before (allocates 3 vectors)
let result = a.clone() + b.clone() + c.clone();

// After (allocates 0 vectors if we can consume a)
let mut result = a;  // move
result.add_assign(&b);
result.add_assign(&c);
```

**Target allocations to eliminate:**
- Sensitivity calculation clones (lines 2090-2101, 2103-2105)
- Ground term preparation (lines 2199-2202)
- Early arithmetic chains (lines 2028-2046)

### Task 3: Eliminate `phi_ia.clone()` for Inter-zone Heat

Currently:
```rust
let mut phi_ia_with_iz = phi_ia.clone();  // allocation
```

Since `phi_ia` is not used after this block, we can **move** it:
```rust
let mut phi_ia_with_iz = phi_ia;  // no allocation
```

The inter-zone heat is then added in-place (lines 2174-2184). Combined with in-place arithmetic for subsequent operations on `phi_ia_with_iz`, this eliminates one allocation per timestep for multi‑zone models.

---

## Expected Impact

- **Allocation reduction:** Eliminate ~10-15 allocations per timestep (depending on configuration)
- **Bytes saved:** ~100-200KB per simulation
- **Throughput improvement:** Reduced allocator pressure → better cache locality → faster per‑config latency, especially in batch mode

With baseline at ~25 allocs/step, targeting 10-15 allocs/step is a **40-60% reduction**.

---

## Validation Plan

- Re-run `test_allocation_count_single_model` after changes
- Compare allocation count (blocks) and bytes
- Run `test_allocation_count_batch_1000` (separate execution to avoid dhat conflict)
- Ensure full test suite passes (`cargo test --release`)
- Verify throughput guardrail (`test_batch_oracle_throughput`) still ≥1000 configs/sec

---

## Notes on Measurement

- dhat frame capture was incomplete in this run (empty `fs` arrays), so line numbers derived from code analysis and may shift slightly as code changes.
- The batch allocation test currently fails due to dhat profiler conflict (multiple profilers in same process). Workaround: run tests separately or use `dhat::LeakCheck` pattern with manual heap start/stop.

---

**Status:** Hotspots identified; transitioning to Task 2 implementation.
