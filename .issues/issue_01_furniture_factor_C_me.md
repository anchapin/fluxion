## Problem

Current `C_me` (internal thermal capacitance) is calculated as `total_cap * (1.0 - envelope_mass_fraction)` which gives ~3.0e6 J/K for Case 900. This produces a time constant that's too slow (~38 hours).

## Research Baseline

From `research_internal_mass_capacitance.md`:
- EnergyPlus default: `55,000 J/m²K` for furniture thermal mass
- Recommended formula: `C_me = A_floor × 55,000 × f_furniture`
- For Case 900 (48 m² floor, commercial f_furniture=0.5): `C_me = 48 × 55,000 × 0.5 = 1.32e6 J/K`

## Resolution

**Implement furniture factor-based C_me calculation in `from_spec()`**:

```rust
let furniture_factor = match spec.building_type {
    BuildingType::Residential => 0.3,
    BuildingType::Commercial => 0.5,
    BuildingType::Institutional => 0.5,
};
let c_me = zone_floor_area * 55_000.0 * furniture_factor;
```

## Expected Values

| Case | A_floor (m²) | f_furniture | C_me (J/K) | Current (J/K) |
|------|-------------|-------------|------------|---------------|
| Case 600 | 48 | 0.3 | 7.92e5 | ~6.0e5 |
| Case 900 | 48 | 0.5 | 1.32e6 | ~3.0e6 |

## Tasks

- [ ] Add `BuildingType` enum to `CaseSpec` (see Issue 3)
- [ ] Update `from_spec()` in `thermal_model_core.rs` to compute `C_me` from furniture factor
- [ ] Update test `test_internal_mass_capacitance_values` to verify new C_me values
- [ ] Run ASHRAE 140 validation tests

## Files to Modify

- `src/sim/thermal_model_core.rs` - `from_spec()` function
- `src/sim/case_spec.rs` - Add `BuildingType` enum

## Reference

- research_internal_mass_capacitance.md Section 5 (Recommended Formula)
- research_internal_mass_capacitance.md Section 6.1 (Implementation Recommendations)
