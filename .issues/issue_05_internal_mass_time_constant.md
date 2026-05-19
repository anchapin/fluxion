## Problem

The time constant τ_me = C_me / h_tr_me should be approximately 3.4 hours for Case 900 (and ~3.4 hours for Case 600 as well, since both C_me and h_tr_me scale with furniture factor).

This is a key validation point: the physics-based calculation produces a time constant that is independent of furniture factor, which is the correct behavior.

## Research Baseline

From `research_internal_mass_capacitance.md` Section 5:

```
τ_me = C_me / h_tr_me

For Case 900 (f_furniture = 0.5):
- C_me = 1.32e6 J/K
- h_tr_me = 108 W/K
- τ_me = 1.32e6 / 108 = 12,222 seconds ≈ 3.4 hours
```

**Important**: τ_me is independent of f_furniture because both C_me and h_tr_me scale with f_furniture:

| f_furniture | C_me (J/K) | h_tr_me (W/K) | τ_me (hours) |
|-------------|-----------|--------------|--------------|
| 0.2 | 5.28e5 | 43.2 | 3.4 |
| 0.3 | 7.92e5 | 64.8 | 3.4 |
| 0.5 | 1.32e6 | 108 | 3.4 |

## Resolution

Add test to verify this invariant:

```rust
#[test]
fn test_internal_mass_time_constant_invariant() {
    // τ_me should be ~3.4 hours regardless of furniture factor
    // because both C_me and h_tr_me scale with f_furniture

    let tau_me_expected_hours = 3.4;
    let tolerance_hours = 0.5;  // ±0.5 hour tolerance

    for f_furniture in [0.2, 0.3, 0.5] {
        let c_me = floor_area * 55_000.0 * f_furniture;
        let h_tr_me = 4.5 * f_furniture * floor_area;
        let tau_me_hours = c_me / h_tr_me / 3600.0;

        assert!(
            (tau_me_hours - tau_me_expected_hours).abs() < tolerance_hours,
            "τ_me = {} hours for f_furniture={}, expected ~{}",
            tau_me_hours, f_furniture, tau_me_expected_hours
        );
    }
}
```

## Tasks

- [ ] Add `test_internal_mass_time_constant_invariant` to thermal model tests
- [ ] Verify τ_me ≈ 3.4 hours for Case 900 (floor_area=48 m²)
- [ ] Verify τ_me ≈ 3.4 hours for Case 600 (floor_area=48 m²)
- [ ] Verify τ_me is truly invariant across furniture factors (0.2, 0.3, 0.5)
- [ ] Test that envelope mass time constant τ_em is different (~8 hours) and dominates overall behavior

## Files to Modify

- `tests/test_6r2c_comprehensive.rs` or thermal model test file

## Reference

- research_internal_mass_capacitance.md Section 5 (Time Constant Analysis)
- research_internal_mass_capacitance.md Section 6.4 (Time Constant Analysis table)
