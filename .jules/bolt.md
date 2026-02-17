## 2025-02-12 - [Guarding ContinuousTensor Operations]
**Learning:** The `ContinuousTensor` abstraction (implemented by `VectorField`) allocates new `Vec`s for every arithmetic operation. This becomes a bottleneck in hot loops like `step_physics`.
**Action:** Guard tensor operations with scalar checks (e.g., `if coefficient.abs() > 1e-9 { ... } else { ... }`) to skip allocation chains for zero-valued coefficients (like inter-zone conductance or thermal bridges).
