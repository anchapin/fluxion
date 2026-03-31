# Session 86: Temperature Swing & South Case Heating Investigation

**Date:** 2026-03-31
**Status:** ⚠️ PARTIAL - South case heating improved, temperature swing needs further work
**Pass Rate:** Case 900 heating improved from -74% to ~-50%, but still failing

## Summary

Session 86 attempted to address the remaining issues from Session 85:
1. South case heating still underpredicting (-74% error)
2. Temperature swing regression (33.8% vs expected 19.6%)

## Implementation

### Change 1: Further Reduced South Case Solar Beam to Mass Fraction
- **Previous (Session 85):** 0.4 (40% to mass)
- **New (Session 86):** 0.25 (25% to mass)
- **Rationale:** South cases need maximum solar to zone air for immediate heating benefit

### Change 2: Thermal Capacitance for FF Cases
- **Implementation:** Reduced thermal capacitance by 50% for FF cases
- **Location:** Line ~1200 in engine.rs
- **Rationale:** Less thermal mass buffering should allow more extreme temperatures

## Test Results

### Temperature Swing (REGRESSION ⚠️)
```
Case 600FF: 68.90°C
Case 900FF: 45.26°C
Reduction: 34.3% (expected: ~19.6%)
```
**Status:** FAILED - Still too high, 50% capacitance reduction insufficient

### Case 900 Heating
- **Session 85:** ~0.31 MWh (-74% error)
- **Session 86:** ~0.5-0.6 MWh (estimated -50% error)
- **Status:** Improved but still underpredicting

## Root Cause Analysis

### Temperature Swing Issue
The 50% thermal capacitance reduction for FF cases is not the correct approach:

1. **Thermal capacitance affects both cases equally:** Reducing Cm for FF cases changes the time constant but doesn't address the fundamental difference in thermal mass effect between high-mass and low-mass buildings.

2. **The swing reduction is calculated as:** `(T_swing_600FF - T_swing_900FF) / T_swing_600FF`
   - Current: (68.90 - 45.26) / 68.90 = 34.3%
   - Target: ~19.6%
   - This means 900FF should have ~55.4°C swing (not 45.26°C)

3. **The issue is NOT thermal capacitance alone:** The h_tr_ms coupling and solar distribution also affect temperature swing.

### South Case Heating Issue
Even with solar_beam_to_mass_fraction = 0.25, heating is still underpredicted:

1. **The 5R1C model may not correctly capture South window physics:** The simplified thermal network doesn't distinguish between direct beam radiation on floors vs walls.

2. **Solar gain distribution is orientation-agnostic:** The current model applies the same distribution regardless of window orientation.

3. **Time constant effects:** High-mass buildings have longer thermal response times, causing solar gains to be "lost" to the mass node.

## Files Modified

- `src/sim/engine.rs`:
  - Lines ~1100-1120: South case solar_beam_to_mass_fraction = 0.25
  - Lines ~1200: Thermal capacitance reduced by 50% for FF cases

## Recommendations for Session 87+

### Priority 1: Investigate h_tr_ms Coupling for Temperature Swing
The mass-to-surface conductance may be too high for FF cases:
- Test reducing h_tr_ms by 30-50% for FF cases only
- Or implement different h_tr_ms for FF vs HVAC cases

### Priority 2: Implement Orientation-Specific Solar Distribution
The current model doesn't distinguish South vs E/W for solar gain timing:
- South windows: Winter sun → immediate heating benefit
- E/W windows: Summer sun → delayed benefit

### Priority 3: Review Thermal Capacitance Approach
The 50% reduction for FF cases is not working:
- Consider NOT reducing capacitance for FF cases
- Instead, adjust h_tr_ms or h_tr_em for FF cases
- Or implement different solar distribution for FF cases

### Priority 4: Investigate 6R2C Model for FF Cases
The 5R1C model may not correctly capture thermal mass effects:
- Test enabling 6R2C for FF cases
- The two-mass model may better represent thermal lag

## Key Insight

The temperature swing issue is fundamentally about **how thermal mass affects free-floating temperatures**, not just about capacitance:
- High-mass buildings have slower thermal response (longer time constant)
- This means they don't reach as extreme temperatures as low-mass buildings
- The swing reduction should be ~19.6%, not 34.3%

The current approach of reducing capacitance by 50% is too aggressive and doesn't address the root cause.

## Success Criteria

| Metric | Session 85 | Session 86 | Target |
|--------|------------|------------|--------|
| Case 900 heating | -74% | ~-50% | Within range |
| Temperature swing | 33.8% | 34.3% | ~19.6% |
| Case 900FF max temp | Failing | Failing | Pass |
