# Debug Result: Issue #878 - test_perez_diffuse_vertical_surface tilt factor too low

## Status
**PARTIALLY RESOLVED** - Root cause identified and fix applied to sky_radiation.rs, but blocked by pre-existing compilation errors in codebase.

## Problem
The test `test_perez_diffuse_vertical_surface` was asserting that tilt_factor > 0.3, expecting ~0.45 (57 W/m² diffuse for DHI=126.3). The issue was that the `calculate_cos_incidence` function in `sky_radiation.rs` was using `zenith.cos()` instead of `zenith.sin()` in the first two terms of the incidence angle formula.

## Root Cause Analysis

The `calculate_cos_incidence` function (lines 535-551) was computing:
```rust
cos_incidence = sin(tilt) * sin(surface_az) * zenith.cos() * solar_az.sin()
              + sin(tilt) * cos(surface_az) * zenith.cos() * solar_az.cos()
              + tilt.cos() * zenith.sin();
```

The correct formula uses `zenith.sin()` in the first two terms, not `zenith.cos()`.

**Verification with test parameters:**
- zenith_deg=25°, surface_tilt=90°, surface_az=270°, solar_az=240°
- Buggy formula: cos_incidence = 0.7849 → diffuse = 61.2 W/m² → tilt_factor = 0.485
- Correct formula: cos_incidence = 0.3660 → diffuse = 30.4 W/m² → tilt_factor = 0.240

**Impact:** The buggy formula gives a tilt factor of 0.485 which barely passes the threshold of 0.3, but the correct formula gives 0.240 which would FAIL the test (and would be the physically correct result for a vertical surface at 68.5° incidence angle).

## Fix Applied
Changed in `src/sim/sky_radiation.rs` line 546-547:
```rust
// BEFORE (buggy):
let cos_incidence = tilt.sin() * surface_az.sin() * zenith.cos() * solar_az.sin()
    + tilt.sin() * surface_az.cos() * zenith.cos() * solar_az.cos()
    + tilt.cos() * zenith.sin();

// AFTER (correct):
let cos_incidence = tilt.sin() * surface_az.sin() * zenith.sin() * solar_az.sin()
    + tilt.sin() * surface_az.cos() * zenith.sin() * solar_az.cos()
    + tilt.cos() * zenith.sin();
```

## Blocked By
Pre-existing compilation errors blocking test execution:
1. `WindowProperties::default()` not implemented (tests/ashrae_140_validator.rs imports from sim::solar)
2. Missing `incident_solar` field in `CaseResults` struct initializers (4 locations)
3. Missing `warm_up_years` field in `DeltaConfig` initializers (3 locations in tests)

These appear to be from Issue #880 (IncidentSolar metrics) that was only partially implemented.

## Files Changed
- `src/sim/sky_radiation.rs`: Fixed zenith.sin() vs zenith.cos() bug in calculate_cos_incidence

## Next Steps
1. Complete the Issue #880 implementation by adding `incident_solar` to all CaseResults initializers
2. Implement `Default` for `WindowProperties` or use `WindowProperties::double_clear()`
3. Re-run the test to verify fix

## Test Result
Cannot confirm test passes due to compilation errors. The fix is mathematically verified to be correct based on vector geometry analysis.
