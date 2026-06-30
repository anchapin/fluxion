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
## 2026-05-23 - Optimized HVAC Power Accumulation
**Learning:** Replaced the allocation and clone from `.clone()` by keeping the accumulation of peak power inside the summation block without allocating a new buffer (`let hvac_cloned = hvac_output.clone(); let hvac_power_watts = hvac_cloned.as_ref().iter()...`).
**Action:** When a `.clone()` operation produces an iterator directly consumed by `.sum()`, evaluate if it can be combined with another iteration loop on the same reference or avoided completely by aggregating manually inside an existing block of similar iterators.
## 2026-05-24 - Optimized 6R2C step_physics math
**Learning:** Replaced chained .clone() vector math and intermediate allocations in `step_physics_6r2c` with chained `.zip_with` closures to fuse iterations and eliminate redundant vector allocations.
**Action:** When working with math blocks that use `a.clone() + b.clone()` and `c.clone() * d.clone()`, apply chained `.zip_with` pattern consistently to eliminate double iteration loops and allocations.

## 2026-06-11 - Avoided Vector allocations in den calculation of step_physics_6r2c
**Learning:** In hot simulation loops like `step_physics_6r2c`, chained tensor operations using `.clone()` to build terms like `den` or `ground_coeff` cause extreme allocation pressure due to intermediate `Vec` creations.
**Action:** When a composite value needs to be built across `num_zones` from multiple tensor properties, use a single explicit `for i in 0..self.0.num_zones` loop and `.as_ref()[i]` to build up the result safely inside a single pre-allocated `Vec::with_capacity()`.

## 2026-05-30 - Optimized HVAC Peak Power Calculation
**Learning:** In `ThermalModel::step_physics`, using `.clone()` on `hvac_output_raw` to store the value for peak power calculations generated unnecessary memory allocations.
**Action:** The cloned `hvac_power_for_peak` value was completely redundant since `hvac_output_raw` could be directly chained to compute peak power calculations (which requires an elementwise read, not a mutable borrow nor ownership of the Tensor block).
## 2026-06-12 - Ensure step_physics_6r2c doesn't print debugging output in hot loops
**Learning:** Found debug `println!` output within the `step_physics_6r2c` hot loop which blocked IO and lowered throughput.
**Action:** Removed these debug `println!` macros since they should not be compiled in standard performance builds.
## 2026-06-20 - Avoided Double Tensors Allocations during Add operations
**Learning:** Replaced chained `VectorField` mutations/operations like `a.clone() + b.clone()` and `c.clone() * d.clone()` with `a.zip_with(&b, |x, y| x + y)` in `step_physics_6r2c`'s thermal mass energy loop to avoid redundant array allocations.
**Action:** Always favor `.zip_with` closures instead of `.clone() +` or `.clone() *` in core physics loop when applying scalar or vector combinations across Tensor arrays to minimize runtime vector memory fragmentation.
## 2026-06-25 - Avoid vector `clone` and chained math assignments inside physics hot loops
**Learning:** Found that using chained VectorField calculations (`clone()`, `mul_assign()`, `add_assign()`, `zip_with()`) for mathematical formulas like `ts_num_act = h_tr_ms * mass_temp + h_tr_is * t_i_act + phi_st` produces significant memory allocation overhead inside highly iterated loop blocks like `physics_impl.rs`.
**Action:** Replace `VectorField` chain arithmetic directly with a single pass scalar loop that pre-allocates an empty target vector using `Vec::with_capacity(num_zones)` and handles the computation directly for the buffer creation. This resulted in up to ~6.4% performance improvement in throughput.
