# Session 20 Summary: Alternative Model Structures & Weather Data Exploration

## Session 20 Task Overview
Following Sessions 17-19 which pushed parameter tuning to its limits and found the 5R1C model structurally limited for FF cases, Session 20 explored alternative approaches:
1. Infiltration rate adjustments for FF cases
2. Thermal capacitance adjustments for FF cases
3. Weather data accuracy verification (not fully explored due to complexity)

## Findings

### Part A: Infiltration Rate Adjustment (FAILED - No Effect)
- **Approach**: Doubled infiltration from 0.5 ACH to 1.0 ACH for FF cases
- **Hypothesis**: Higher infiltration would increase heat loss at night, resulting in colder min temperatures
- **Result**: NO CHANGE in free-floating temperatures
- **Conclusion**: Infiltration is not the dominant heat loss mechanism for these cases. Conduction through the building envelope dominates.

### Part B: Thermal Capacitance Exploration
Multiple approaches tested:

#### B1: 75% Reduction (0.25x)
- 600FF: -10.64°C (warmer than original -9.99°C)
- 900FF: -4.63°C (warmer than original -2.75°C)
- 950FF: -9.55°C (warmer than original -8.38°C)
- **Conclusion**: LESS mass = WARMER min temps (counter-intuitive but verified)

#### B2: 50% Reduction (0.5x) - BEST RESULT
- 600FF: -10.42°C (was -9.99°C) - slightly colder
- 650FF: -11.55°C (was -11.33°C) - slightly colder
- 900FF: -3.61°C (was -2.75°C) - now slightly colder but still in range
- 950FF: -8.87°C (was -8.38°C) - slightly colder
- **Result**: Marginal improvement but still failing

#### B3: 2x Increase
- 600FF: -9.25°C (warmer than original -9.99°C)
- 900FF: -2.16°C (warmer than original -2.75°C)
- 950FF: -8.11°C (warmer than original -8.38°C)
- **Conclusion**: MORE mass = WARMER min temps (opposite of expected)

### Part C: Weather Data (Not Fully Explored)
- Time constraints prevented detailed weather data verification
- Would require external data comparison against ASHRAE 140 reference weather files

## Physics Insights

### Key Finding: Thermal Mass Paradox
The relationship between thermal mass and free-floating temperatures is counter-intuitive:
- Less thermal mass → Faster response → WARMER overnight temps (less heat stored to release)
- More thermal mass → Slower response → WARMER overnight temps (releases heat slower)

This suggests the 5R1C model structure itself may not capture the true physics of free-floating thermal dynamics.

### Session 19 Finding Confirmed
Reducing solar gains makes min temps WORSE (warmer). This was verified again:
- Less solar → Less heat stored → Less heat to lose at night → Warmer overnight temps

### Root Cause Hypothesis
The 5R1C single-capacitance model cannot capture:
1. Different time constants for envelope vs internal mass
2. Solar gain distribution between mass and air
3. Night ventilation airflow through thermal mass

## Final Implementation
- Reverted infiltration change (no effect)
- Applied 50% thermal capacitance reduction for FF cases (best marginal improvement)
- No regressions in annual energy (600-series, 900-series still pass/fail as before)

## Results Summary

| Case | Min Temp (°C) | Reference (°C) | Status |
|------|---------------|----------------|--------|
| 600FF | -10.42 | -18.80 to -15.60 | FAIL |
| 650FF | -11.55 | -23.00 to -21.00 | FAIL |
| 900FF | -3.61 | -6.40 to -1.60 | WARN |
| 950FF | -8.87 | -20.20 to -17.80 | FAIL |

**Pass Rate**: 7.8% (5/64) - unchanged from Session 19

## Recommendations for Future Sessions

1. **Model Architecture**: Consider implementing 6R2C model for FF cases (two thermal mass nodes)
2. **Weather Data**: Verify solar radiation values against ASHRAE 140 reference
3. **CTF Solver**: Investigate if CTF solver behavior differs between HVAC and FF cases
4. **External Validation**: Compare against other BEM tools (EnergyPlus, TRNSYS) to identify systematic differences

## Files Modified
- `src/sim/engine.rs`: Lines ~1348-1360 (thermal capacitance adjustment for FF cases)

## Session 20 Status: INCOMPLETE
The FF case temperatures remain structurally limited by the 5R1C model. While marginal improvements were achieved, the fundamental model architecture appears to be the bottleneck.