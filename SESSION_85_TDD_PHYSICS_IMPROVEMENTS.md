# Session 85: TDD Physics Improvements - Orientation-Dependent Solar Distribution

**Date:** 2026-03-31
**Status:** ⚠️ PARTIAL - E/W cases fixed, South cases need further work
**Pass Rate:** ~14% (raw, without empirical corrections)

## Summary

Session 85 implemented a physics-based fix for the contradictory heating behavior between South and E/W window cases by making the solar beam distribution orientation-dependent.

## Implementation

Modified `solar_beam_to_mass_fraction` based on window orientation:
- **South windows (900, 910, 940, 950):** 0.4 (40% to mass) - REDUCED from 0.7
- **E/W windows (920, 930):** 0.5 (50% to mass) - unchanged from Session 84
- **Sunspace (960):** 0.4 (40% to mass) - unchanged
- **Low-mass (600 series):** 0.3 (30% to mass) - unchanged

## Physics Rationale

**South Cases (900, 910, 940, 950):**
- Previous 0.7 fraction sent too much solar to thermal mass
- Stored heat doesn't provide immediate heating benefit during winter
- Reducing to 0.4 sends more solar directly to zone air for immediate heating
- Expected: Heating increases from ~0.37 MWh to ~1.5 MWh (target: 1.17-2.04 MWh)

**E/W Cases (920, 930):**
- Morning/evening sun hits vertical walls, not floor
- 0.5 fraction (50% to mass) correctly captures reduced mass coupling
- Session 84 fix already addressed E/W overprediction

## Test Results

### E/W Cases (PASSING ✅)
| Case | Heating (MWh) | Reference | Status |
|------|---------------|-----------|--------|
| 920 | ~4.24 | 3.26-4.30 | ✅ PASS |
| 930 | ~5.27 | 4.14-5.34 | ✅ PASS |

### South Cases (NEEDS WORK ❌)
| Case | Heating (MWh) | Reference | Error |
|------|---------------|-----------|-------|
| 900 | ~0.31 | 1.17-2.04 | -74% |
| 910 | ~0.37 | 1.51-2.28 | -76% |
| 940 | ~0.21 | 0.79-1.41 | -75% |

### Temperature Swing (REGRESSION ⚠️)
- Case 900FF swing reduction: 33.8% (expected: ~19.6%)
- The reduced solar_beam_to_mass_fraction affects free-floating behavior

## Root Cause Analysis

The South case underprediction persists because:

1. **Solar distribution alone is insufficient** - The 0.4 fraction helps but doesn't fully address the fundamental thermal network dynamics

2. **Thermal mass coupling (h_tr_ms) may be too aggressive** - Heat stored in mass during day is released too slowly to benefit zone air temperature

3. **Time constant effects** - High-mass buildings have longer thermal response times, causing solar gains to be "lost" to the mass node

4. **Zone air sensitivity** - The sensitivity calculation may not properly account for orientation-specific solar gain patterns

## Files Modified

- `src/sim/engine.rs` (lines ~1520-1560): Updated solar_beam_to_mass_fraction logic with SESSION 85 documentation

## Recommendations for Session 86+

### Priority 1: Further Reduce South Case solar_beam_to_mass_fraction
Try reducing from 0.4 to 0.2-0.3:
```rust
if has_south_windows && !has_ew_windows {
    0.25  // Even more solar to zone air for immediate heating
}
```

### Priority 2: Investigate h_tr_ms Coupling
The mass-to-surface conductance may be too high, causing heat to drain from zone air too quickly:
- Test reducing h_tr_ms by 30-50% for South cases
- Or implement mode-specific h_tr_ms (lower in heating season)

### Priority 3: Seasonal Solar Distribution
Implement seasonal variation in solar_beam_to_mass_fraction:
- Winter (heating season): Lower fraction (more to air)
- Summer (cooling season): Higher fraction (more to mass)

### Priority 4: Fix Temperature Swing Regression
The 33.8% swing reduction (vs 19.6% expected) needs correction:
- May need to adjust thermal capacitance for FF cases
- Or implement different solar distribution for FF vs HVAC cases

## Key Insight

The contradictory behavior (South underpredicting, E/W overpredicting) is fundamentally about **when** solar gains benefit the zone:
- **South windows**: Winter sun provides heat when needed → need immediate benefit (low mass fraction)
- **E/W windows**: Summer sun causes overheating → need delayed benefit (higher mass fraction)

The current uniform approach doesn't capture this seasonal/orientation interaction.

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Case 900 heating | ~0.31 MWh | 1.17-2.04 MWh |
| Case 920 heating | ~4.24 MWh | 3.26-4.30 MWh ✅ |
| Case 930 heating | ~5.27 MWh | 4.14-5.34 MWh ✅ |
| Temperature swing | 33.8% | ~19.6% |
| Overall pass rate | ~14% | >40% |
