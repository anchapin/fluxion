## 2026-05-14 - Optimized hvac_power_demand vector math
**Learning:** Replaced explicit .clone() and * multiplication with zip_with for elementwise Tensor operations.
**Action:** Always use zip_with for Tensor arithmetic instead of cloning buffers when possible.
## 2026-05-18 - Optimized tensor arithmetic in thermal mass update
**Learning:** Avoid using  followed by  or  for chained element-wise  operations in hot loops. This allocates unnecessary intermediate buffers. Instead,  should be used which correctly fuses operations and yields significant performance improvements (~35-50% in  benchmarks).
**Action:** Always favor  and iterator logic over chained mutator logic that depends on cloned tensors when implementing math inside simulation steps.

## 2026-05-18 - Optimized tensor arithmetic in thermal mass update
**Learning:** Avoid using `.clone()` followed by `.mul_assign()` or `.add_assign()` for chained element-wise `ContinuousTensor` operations in hot loops. This allocates unnecessary intermediate buffers. Instead, `.zip_with` should be used which correctly fuses operations and yields significant performance improvements (~35-50% in `solve_timesteps_1year_10zones` benchmarks).
**Action:** Always favor `.zip_with` and iterator logic over chained mutator logic that depends on cloned tensors when implementing math inside simulation steps.
