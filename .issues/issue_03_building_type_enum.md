## Problem

Need a way to select appropriate furniture factor based on building type (residential vs commercial/institutional). Currently no building type classification exists in `CaseSpec`.

## Research Baseline

From `research_internal_mass_capacitance.md`:

| Building Type | f_furniture | C_me factor | h_tr_me factor |
|---------------|-------------|-------------|----------------|
| Residential | 0.3 | 0.3 × A_floor | 0.3 × A_floor |
| Commercial | 0.5 | 0.5 × A_floor | 0.5 × A_floor |
| Institutional | 0.5 | 0.5 × A_floor | 0.5 × A_floor |

## Resolution

**Add `BuildingType` enum to `CaseSpec`**:

```rust
/// Building usage type for thermal mass calculations
pub enum BuildingType {
    /// Residential buildings - lighter furniture, f_furniture = 0.3
    Residential,
    /// Commercial buildings - heavier furniture, f_furniture = 0.5
    Commercial,
    /// Institutional buildings (schools, hospitals) - f_furniture = 0.5
    Institutional,
}

impl Default for BuildingType {
    fn default() -> Self {
        BuildingType::Residential
    }
}
```

Add to `CaseSpec`:
```rust
pub struct CaseSpec {
    // ... existing fields ...
    pub building_type: BuildingType,
}
```

## Tasks

- [ ] Add `BuildingType` enum to `case_spec.rs`
- [ ] Add `building_type: BuildingType` field to `CaseSpec`
- [ ] Set default `BuildingType::Residential` for backward compatibility
- [ ] Update all test cases that create `CaseSpec` to include building_type
- [ ] Document mapping in `case_spec.rs` doc comments

## Files to Modify

- `src/sim/case_spec.rs` - Add enum and field

## Reference

- research_internal_mass_capacitance.md Section 5 (furniture factor table)
- research_internal_mass_capacitance.md Section 6.3 (Building Type Configuration)
