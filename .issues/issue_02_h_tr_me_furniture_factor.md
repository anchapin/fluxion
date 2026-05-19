## Problem

Current `h_tr_me` (conductance: envelope mass → internal mass) is calculated as `h_ms × 0.1 × A_floor` giving ~21.6 W/K for Case 900. This is too low, causing envelope and internal mass to behave too independently.

## Research Baseline

From `research_internal_mass_capacitance.md`:
- Formula: `h_tr_me = h_ms × A_int` where `h_ms = 4.5 W/(m²·K)`
- `A_int = f_furniture × A_floor`
- For Case 900 (f_furniture=0.5): `h_tr_me = 4.5 × 0.5 × 48 = 108 W/K`

This gives a time constant τ_me = C_me/h_tr_me ≈ 3.4 hours, consistent with furniture thermal mass behavior.

## Resolution

**Update h_tr_me calculation in `from_spec()`**:

```rust
let furniture_factor = 0.5;  // Could be made configurable per building type
let a_int = furniture_factor * zone_floor_area;
let h_ms = 4.5;  // Interior surface convection coefficient W/(m²·K)
let h_tr_me = h_ms * a_int;
```

## Expected Values

| Case | A_floor (m²) | f_furniture | A_int (m²) | h_tr_me (W/K) | Current (W/K) |
|------|-------------|-------------|-----------|---------------|---------------|
| Case 600 | 48 | 0.3 | 14.4 | 64.8 | ~21.6 |
| Case 900 | 48 | 0.5 | 24.0 | 108.0 | ~21.6 |

## Tasks

- [ ] Update `h_tr_me` calculation in `thermal_model_core.rs:1119-1131`
- [ ] Change from `0.1 * zone_floor_area` to `furniture_factor * zone_floor_area`
- [ ] Update test `test_h_tr_me_calculation` to verify new values
- [ ] Verify time constant τ_me ≈ 3.4 hours (see Issue 5)

## Files to Modify

- `src/sim/thermal_model_core.rs` - `from_spec()` h_tr_me calculation

## Reference

- research_internal_mass_capacitance.md Section 5.2 (Recommended h_tr_me)
- Issue 692 (supersedes previous hardcoded 100 W/K approach)
