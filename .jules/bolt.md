## 2024-05-23 - Performance Optimization via Reference Arithmetic
**Learning:** By adding `Add<&Self>` (and other ops) to the `ContinuousTensor` trait, we can avoid deep cloning of vectors in the core physics loop (`step_physics`). This resulted in an ~18% performance improvement (23.6ms -> 19.3ms). Also, we used a `TensorCow` pattern to handle conditionally owned/borrowed tensors.
**Action:** When working with heavy types in hot loops, ensure arithmetic traits support references to avoid unnecessary allocations.
