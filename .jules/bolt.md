## 2026-05-14 - Optimized hvac_power_demand vector math
**Learning:** Replaced explicit .clone() and * multiplication with zip_with for elementwise Tensor operations.
**Action:** Always use zip_with for Tensor arithmetic instead of cloning buffers when possible.

## 2026-05-17 - Avoided intermediate Vector allocations in hvac_demand_from_ideal_loads
**Learning:** Eliminating intermediate `Vec` allocations (`vec![val; num_zones]`) used solely for slice creation significantly reduces hot-loop overhead. By resolving parameters per-element directly in the loop, multiple O(N) allocations were replaced with a single `Vec::with_capacity(N)` and scalar arithmetic.
**Action:** When a method takes slices of uniform values just to perform element-wise operations, inline the constant value lookups into the iteration loop to bypass building intermediate vectors entirely.

## 2026-05-18 - Optimized tensor arithmetic in thermal mass update
**Learning:** Avoid using `.clone()` followed by `.mul_assign()` or `.add_assign()` for chained element-wise `ContinuousTensor` operations in hot loops. This allocates unnecessary intermediate buffers. Instead, `.zip_with` should be used which correctly fuses operations and yields significant performance improvements (~35-50% in `solve_timesteps_1year_10zones` benchmarks).
**Action:** Always favor `.zip_with` and iterator logic over chained mutator logic that depends on cloned tensors when implementing math inside simulation steps.

## 2026-05-19 - Avoided intermediate Vector allocations in hvac_demand_from_ideal_loads (Take 2)
**Learning:** Slices of uniform values can be created from single variables as slices rather than by allocating memory dynamically.
**Action:** Replaced `vec![val; N]` with `&[val]` when the loop logic passes only single element slices down.
