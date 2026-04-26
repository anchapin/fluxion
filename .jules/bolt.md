## 2024-04-26 - Prevent redundant sol-air temperature calculation in hot loop
**Learning:** `step_physics_5r1c` and `step_physics_6r2c` were re-calculating the sol-air temperature (`t_sol_air`) in a local loop despite it already being returned by the preceding `prepare_solvers_and_sol_air` call. This caused redundant `Vec::with_capacity` allocations and `O(N)` calculations per timestep.
**Action:** When working on physics steps, always check if helpers like `prepare_solvers_and_sol_air` or cached `derived_` fields already provide the needed vectors to avoid double-allocation.
