## 2026-04-07 - [HVAC Power Demand Optimization]
**Learning:** In hot loops like `ThermalModel::hvac_power_demand`, early returning for disabled zones using `continue` (instead of performing calculations and then multiplying by a mask in a second pass) eliminates unnecessary allocations and arithmetic.
**Action:** Use early returns inside mathematical `VectorField` calculations rather than masking later, particularly when iterating over per-zone settings.
