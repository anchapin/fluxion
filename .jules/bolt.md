## 2026-05-14 - Optimized hvac_power_demand vector math
**Learning:** Replaced explicit .clone() and * multiplication with zip_with for elementwise Tensor operations.
**Action:** Always use zip_with for Tensor arithmetic instead of cloning buffers when possible.
