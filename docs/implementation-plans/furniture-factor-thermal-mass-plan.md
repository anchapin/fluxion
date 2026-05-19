# Plan: Furniture Factor Thermal Mass Issues (Issues 1-5)

**Last Updated**: 2026-05-19
**Branch**: `fix/furniture-factor-thermal-mass-issues-1-2-3-5`
**Status**: Implementation complete, pending merge and validation

---

## Executive Summary

The furniture factor thermal mass implementation has been **completed on the feature branch** but **not yet merged to main**. The implementation addresses Issues 1-5 related to thermal mass calculations using furniture factors.

---

## Issue Status

| Issue | Title | Status | Notes |
|-------|-------|--------|-------|
| #3 | Add `BuildingType` enum to `CaseSpec` | ✅ Complete | Defined in `ashrae_140_cases.rs` |
| #1 | Furniture factor-based C_me calculation | ✅ Complete | `C_me = A_floor × 55,000 × f_furniture` |
| #2 | Update h_tr_me calculation | ✅ Complete | `h_tr_me = 4.5 × f_furniture × A_floor` |
| #5 | Time constant invariant test | ✅ Complete | `test_internal_mass_time_constant_invariant` added |
| #4 | ASHRAE 140 validation | ⏳ Pending | Requires merge + test run |

---

## Implementation Details

### BuildingType Enum (Issue 3)
**Location**: `src/validation/ashrae_140_cases.rs`

```rust
pub enum BuildingType {
    Residential,   // f_furniture = 0.3
    Commercial,    // f_furniture = 0.5
    Institutional, // f_furniture = 0.5
}
```

### Furniture Factor Formula (Issues 1 & 2)
**Location**: `src/sim/thermal_model_core.rs`

```rust
let furniture_factor = match spec.building_type {
    crate::validation::ashrae_140_cases::BuildingType::Residential => 0.3,
    crate::validation::ashrae_140_cases::BuildingType::Commercial => 0.5,
    crate::validation::ashrae_140_cases::BuildingType::Institutional => 0.5,
};

// C_me = A_floor × 55,000 × f_furniture (J/K)
let c_me = zone_floor_area * 55_000.0 * furniture_factor;

// h_tr_me = 4.5 × f_furniture × A_floor (W/K)
let a_int = furniture_factor * zone_floor_area;
let h_tr_me = h_ms * a_int;  // h_ms = 4.5 W/(m²·K)
```

### Expected Values Verification

| Case | A_floor | f_furniture | C_me (J/K) | h_tr_me (W/K) | τ_me (hours) |
|------|---------|-------------|------------|---------------|--------------|
| Case 600 | 48 m² | 0.3 | 792,000 | 64.8 | 3.4 |
| Case 900 | 48 m² | 0.5 | 1,320,000 | 108.0 | 3.4 |

**Time constant invariant verified**: τ_me ≈ 3.4 hours regardless of furniture factor (since both C_me and h_tr_me scale with f_furniture).

---

## Files Modified

| File | Changes |
|------|---------|
| `src/validation/ashrae_140_cases.rs` | +46 lines (BuildingType enum, field in CaseSpec) |
| `src/sim/thermal_model_core.rs` | +38 lines (furniture factor formula) |
| `src/sim/thermal_model_solvers.rs` | +4 lines (import fix) |
| `tests/test_6r2c_comprehensive.rs` | +76 lines (updated tests) |
| `tests/adaptive_timestep_integration.rs` | +22 lines (updated tests) |

---

## Remaining Tasks

### 1. Merge to Main (High Priority)
```bash
git checkout main
git merge fix/furniture-factor-thermal-mass-issues-1-2-3-5
```

### 2. Run ASHRAE 140 Validation (High Priority)
```bash
cargo test ashrae140
```
Verify:
- Case 600 heating/cooling loads within ±10% of reference
- Case 900 heating/cooling loads within ±10% of reference
- Internal mass temperature shows faster response than envelope mass
- τ_me ≈ 3.4 hours

### 3. Update Issue Files (Low Priority)
The issue files specify `case_spec.rs` for BuildingType, but implementation used `ashrae_140_cases.rs`. This is actually correct since BuildingType is used for ASHRAE case configuration.

---

## Key Finding: Issue Location Discrepancy

The issue files (.issues/) specify:
- `src/sim/case_spec.rs` - Add BuildingType enum

Actual implementation:
- `src/validation/ashrae_140_cases.rs` - BuildingType defined here, used in CaseSpec indirectly

**Rationale**: BuildingType is semantically correct in ashrae_140_cases.rs since it's used for ASHRAE 140 test case configuration. The implementation is architecturally correct.

---

## References

- Issue files: `.issues/issue_0{1,2,3,4,5}_*.md`
- Research: `research_internal_mass_capacitance.md`
- Commit history: `git log origin/fix/furniture-factor-thermal-mass-issues-1-2-3-5`
