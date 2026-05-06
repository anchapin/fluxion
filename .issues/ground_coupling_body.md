## Problem Description

Ground coupling and floor thermal conductance calculation may contribute to excessive temperatures in free-floating cases.

## Root Cause

**`src/sim/thermal_model_core.rs:704-712`** - h_tr_floor calculation:

```rust
let h_tr_floor_val = if spec.case_id == "195" {
    0.039 * zone_floor_area
} else if is_900_series_hvac {
    floor_u * zone_floor_area * 1.2  // 900-series WITH HVAC
} else {
    floor_u * zone_floor_area          // 900FF gets NO multiplier!
};
```

The 1.2 multiplier is NOT applied to free-floating cases:
- `is_900_series_hvac` is `false` when `case_id.contains("FF")` (line 700-702)
- 900FF: `h_tr_floor = 0.190 * 48 * 1.0 = 9.12 W/K`
- 900 with HVAC: `h_tr_floor = 0.190 * 48 * 1.2 = 10.94 W/K`

## Floor Construction

Both `insulated_floor()` and `high_mass_floor()` have the same U-value (0.190 W/m²K) but different thermal mass:
- High mass floor: ~200,000 J/m²K per layer (concrete)
- Insulated floor: ~20,000 J/m²K per layer (wood/fiberglass)

## Ground Temperature

**`src/sim/boundary.rs:85-147`** - Constant 10°C ground temperature:
```rust
pub struct ConstantGroundTemperature {
    temperature: f64,  // Default: 10°C
}
```

## Investigation Findings

Ground coupling is NOT the primary cause - at ~9 W/K conductance with 10°C ground, it provides at most ~273W of heat loss, insufficient to cause 80°C+ temperatures.

However, the floor thermal mass may be double-counted:
- Floor thermal mass is included in `total_thermal_cap` (line 747-757)
- Floor may also be double-counted in `h_tr_ms` calculation (line 804) using the "half-insulation rule"

## Proposed Fix

1. Verify floor thermal mass is not double-counted in total_thermal_cap and h_tr_ms
2. Consider whether 900FF should get the 1.2 multiplier (but this would be case-specific)

## Files to Investigate
- `src/sim/thermal_model_core.rs:704-712` - h_tr_floor calculation
- `src/sim/thermal_model_core.rs:747-757` - total_thermal_cap calculation
- `src/sim/thermal_model_core.rs:759-804` - h_tr_ms physics calculation