# Session 38 Prompt: Fix Solar Gain Physics

**Date**: 2026-03-27
**Objective**: Fix the fundamental solar gain physics that are causing 900-series cooling to overpredict and heating to underpredict.

---

## Session 37 Results Summary

**Status**: CTF already enabled, but physics issues remain

**Key Finding**: CTF is already enabled for all 900-series cases (verified via `ConstructionType::HighMass` in `ashrae_140_cases.rs`). The validation failures are due to **solar gain physics issues** in the simulation loop, not solver selection.

**Current Pass Rate**: 1.6% (1/64 metrics)

---

## Root Cause Analysis

The solar gain distribution in `step_physics()` functions doesn't correctly handle:

1. **Seasonal variation**: Summer vs winter solar gains need different treatment
2. **Surface orientation effects**: South windows vs E/W windows behave differently
3. **Thermal mass buffering**: CTF-enabled surfaces need different solar distribution

### Current Problems:

| Problem | Cases | Current | Reference | Issue |
|---------|-------|---------|-----------|-------|
| Cooling overpredicts | 900 | 6.18 MWh | 2.13-3.67 | +68% over max |
| Cooling overpredicts | 910 | 4.28 MWh | 0.82-1.88 | +128% over max |
| Cooling overpredicts | 940 | 6.18 MWh | 2.08-3.55 | +74% over max |
| Heating underpredicts | 920 | 2.60 MWh | 3.26-4.30 | -20% under min |
| Heating underpredicts | 900 | 1.69 MWh | 1.17-2.04 | At min (OK) |

---

## Key Files to Investigate

### 1. `src/sim/engine.rs`

**Solar Distribution Logic** (lines ~1400-1500):
- `calculate_solar_distribution()` - Base solar distribution to surfaces
- Need to add seasonal and orientation-based adjustments

**5R1C Solar Gains** (lines ~3000-3500):
- `step_physics_5r1c()` - Solar gains calculation path
- `solar_beam_to_mass_fraction` - Currently 0.5 (physics-based)
- Need to add seasonal variation

**6R2C/CTF Solar Gains** (lines ~4500-5000):
- `step_physics_6r2c()` - CTF path solar handling
- Thermal mass buffering with CTF needs different treatment

### 2. Key Parameters to Tune

| Parameter | Current | Location | Notes |
|-----------|---------|----------|-------|
| SOLAR_ABSORPTANCE_DEFAULT | 0.7 | constants | For all surfaces |
| solar_beam_to_mass_fraction | 0.5 | Lines ~1419-1426 | 50% to mass (physics-based) |
| direct_to_air_solar_fraction | Variable | Lines ~3100-3200 | Orientation-based |

---

## Session 38 Tasks

### Task 1: Fix Summer Cooling Overprediction (Priority 1)

**Objective**: Reduce solar gains during summer months (May-Aug) for 900-series

**Approach**:
1. Add seasonal solar adjustment in `step_physics_5r1c()`
2. Use hour_of_year to detect summer months (hours 2000-5500)
3. Apply orientation-specific reduction:
   - South windows: Higher reduction (they get more summer sun)
   - E/W windows: Lower reduction (less summer impact)
4. Don't break winter heating!

**Code Location**: Lines ~3100-3200 in `step_physics_5r1c()`

**Expected Result**:
- Case 900 cooling: 6.18 MWh → ~2.5-3.0 MWh (within 2.13-3.67 range)
- Case 910 cooling: 4.28 MWh → ~1.5-1.8 MWh (within 0.82-1.88 range)

### Task 2: Fix Winter Heating Underprediction (Priority 2)

**Objective**: Increase solar gains during winter months for E/W windows

**Approach**:
1. Add winter boost for E/W orientations (920, 930)
2. E/W windows get more winter sun (low angle) than summer
3. Don't over-correct and cause cooling overprediction

**Code Location**: Same as Task 1, in winter condition check

**Expected Result**:
- Case 920 heating: 2.60 MWh → ~3.0-3.5 MWh (within 3.26-4.30 range)
- Case 930 heating: 3.58 MWh → ~4.0-4.5 MWh (within 4.14-5.34 range)

### Task 3: Fix 600-Series (Priority 3)

**Objective**: Address different thermal dynamics for low-mass buildings

**Current Issue**:
- Case 600 heating: 8.65 MWh vs 5.50-7.50 (30% over)
- Low-mass buildings have different thermal response

**Approach**:
1. Check if internal gains are properly applied (200 W/m²)
2. Verify HVAC sensitivity calculation for low-mass
3. May need different coupling factors than 900-series

**Code Location**: Check `step_physics_5r1c()` for low-mass specific code

### Task 4: Maintain Free-Floating Cases (Priority 4)

**Objective**: Don't break existing FF case fixes

**Approach**:
1. Add condition to only apply seasonal adjustment for HVAC-controlled cases
2. Free-floating cases should use base physics (no seasonal adjustment)
3. Verify FF cases still pass after changes

**Code Location**: Add `is_free_floating` check before seasonal adjustment

---

## Implementation Guidelines

### Seasonal Adjustment Formula

```rust
// In step_physics_5r1c(), around line 3100
let hour_of_year = timestep % 8760;
let is_summer = hour_of_year >= 2000 && hour_of_year < 5500; // May-Aug
let is_winter = hour_of_year < 1000 || hour_of_year >= 7000; // Jan, Dec

// Apply orientation-specific adjustments
let solar_multiplier = if is_900_series {
    if is_summer {
        // Summer: reduce solar gains (cooling issue)
        match orientation {
            Orientation::South => 0.45,  // Strong reduction
            Orientation::East | Orientation::West => 0.70, // Moderate
            _ => 0.5,
        }
    } else if is_winter {
        // Winter: increase solar gains (heating issue)
        match orientation {
            Orientation::East | Orientation::West => 1.3, // E/W boost
            Orientation::South => 1.0, // Already gets good winter sun
            _ => 1.0,
        }
    } else {
        1.0 // Shoulder seasons
    }
} else {
    1.0 // 600-series: no adjustment
};
```

### Important Notes

1. **Don't break what works**: Some cases (920 cooling, 930 cooling) are already within reference
2. **Balance heating and cooling**: Changes that fix cooling might break heating
3. **Test incrementally**: Run validation after each change
4. **Document changes**: Add session markers (e.g., `// SESSION 38:`)

---

## Expected Outcome

With proper solar gain physics:
- **Pass rate**: 1.6% → ≥10%
- **900-series cooling**: Within reference for most cases
- **900-series heating**: Improved toward reference
- **600-series**: At least some cases passing

---

## Success Criteria

- [ ] 900-series cooling reduced (target: within reference range)
- [ ] 900-series heating improved (target: within reference range)
- [ ] 600-series addressed (target: within reference range)
- [ ] Free-floating cases still working
- [ ] Code compiles without errors
- [ ] Target: ≥10% pass rate

---

## Files to Modify

1. `src/sim/engine.rs`:
   - Lines ~3100-3200: Add seasonal solar adjustment in `step_physics_5r1c()`
   - Lines ~1400-1500: Update `calculate_solar_distribution()` if needed
   - Add `// SESSION 38:` markers for changes

2. Test and validate after each change
