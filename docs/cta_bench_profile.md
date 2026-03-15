# CTA Benchmark Baseline Profile

**Plan:** 09-03 (VectorField Cache Locality Optimization)
**Date:** 2025-03-12
**Purpose:** Establish baseline performance measurements for VectorField operations before optimization

---

## Benchmark Environment

- **Rust Version:** Stable (edition 2021)
- **Profile:** Release (optimized)
- **Hardware:** 8-core CPU (system dependent)
- **Data Size:** 10,000 elements
- **Iterations:** 3-4 million per benchmark

---

## Baseline Results

Benchmark run on: 2026-03-12

### Raw Vec Map (baseline reference)

```
raw_map
  time:   [1.2635 µs 1.2711 µs 1.2801 µs]
  outliers: 3/100 (1 high mild, 2 high severe)
```

**Interpretation:** Direct iterator map on raw Vec<f64>. This is the reference minimal implementation.

---

### VectorField Map

```
vector_map
  time:   [1.2779 µs 1.2862 µs 1.2958 µs]
  outliers: 9/100 (7 high mild, 2 high severe)
```

**Observation:** VectorField map performance is very close to raw map (~1% difference). The wrapper abstraction has negligible overhead.

---

### NDArray Map

```
ndarray_map
  time:   [1.3163 µs 1.3216 µs 1.3282 µs]
  outliers: 8/100 (3 high mild, 5 high severe)
```

**Observation:** NDArray implementation is ~3-4% slower than raw Vec. This suggests the more complex array shape handling adds minor overhead.

---

## Performance Profiling Notes

### Cache Locality Analysis

**Perf profiling attempted but restricted:**
- System `perf_event_paranoid` setting is 4, which disallows CPU event access for non-privileged users.
- Error: "Access to performance monitoring and observability operations is limited."
- To enable full perf profiling: `sudo sysctl -w kernel.perf_event_paranoid=-1` (or set 1-2 for kernel-only restriction).
- Without perf, cache miss rates are estimated based on code inspection rather than measured.

**Current implementation characteristics:**
- Sequential access patterns in all operations (good for cache locality)
- `gradient()` uses `windows(3)` which creates temporary slice views (potential cache pressure from pointer chasing)
- Arithmetic ops use `iter_mut().zip()` - cache-friendly (streaming stores)
- `map()` and `zip_with()` allocate new vectors - allocation overhead dominant, not cache misses

---

## Hotspot Candidates for Optimization

Based on code review:

1. **`gradient()` method** (lines 209-229 in cta.rs):
   - **Issue:** Uses `self.data.windows(3)` which creates slice objects for each iteration.
   - **Impact:** For large vectors (8760 timesteps), creates ~8,760 3-element slice views.
   - **Opportunity:** Manual loop with index arithmetic eliminates slice allocations completely.
   - **Expected improvement:** 5-10% faster gradient computation, reduced allocation pressure.

2. **`map()` method** (lines 173-180):
   - Creates new Vec via `collect()` - unavoidable for functional API.
   - But can add `#[inline]` hints if closure is small (compiler likely handles this already).
   - No major optimization expected without changing API semantics.

3. **`zip_with()`** (lines 182-195):
   - Similar to map - allocation is the bottleneck, not cache misses.
   - Already uses `zip()` which is optimal for sequential access.
   - No expected improvement without in-place variant.

4. **Arithmetic ops** (`add`, `sub`, `mul`, `div`):
   - Already optimal: in-place modification with `iter_mut().zip()`.
   - Sequential writes enable store buffer optimization.
   - No changes needed.

---

## Allocation Analysis

Current allocation patterns (from code review):

| Method   | Allocations                      | Reason                         |
|----------|----------------------------------|--------------------------------|
| `map`    | 1 new Vec (size n)               | Returns new tensor             |
| `zip_with` | 1 new Vec (size n)            | Returns new tensor             |
| `gradient` | 1 new Vec (size n) + windows | Slice allocations per iteration|
| `add`/etc | 0                                | In-place, reuses LHS buffer    |

**Total per operation (for n=8760):**
- map/zip_with: ~70 KB allocation overhead (f64 * 8760)
- gradient: ~70 KB + ~8,760 slice allocations (tiny but not free)

---

## Success Criteria for 09-03

After implementing optimizations:

- [ ] `gradient()` benchmark shows ≥5% improvement in ns/iter
- [ ] `map()` and `zip_with()` performance neutral (±2% acceptable)
- [ ] No regressions in other benchmarks (raw_map, ndarray_map)
- [ ] All existing tests pass (`cargo test`)
- [ ] Cache miss rate reduced (if perf can be enabled)
- [ ] No new allocations introduced

---

## Next Steps

1. ✅ Baseline captured (this document)
2. Task 2: Optimize `gradient()` using manual sliding window loop
3. Task 3: Add `map_in_place` helper for future optimization
4. Task 4: Validate with cta_bench and full test suite

---

*This profile provides the baseline measurements to track progress of cache locality optimizations in Wave 1 of Phase 9.*
