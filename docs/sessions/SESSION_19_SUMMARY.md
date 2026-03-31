# Session 19 Summary: Solar Gain & Internal Gains Investigation

## Task Overview
Session 19 aimed to investigate solar gains and internal gains for free-floating (FF) cases to improve min/max temperature predictions.

## Current Baseline (Pre-Session 19)

| Case | Min Temp | Target Min | Max Temp | Target Max | Status |
|------|----------|------------|----------|------------|--------|
| 600FF | -9.99°C | -18.80°C | 41.56°C | 64.90-75.10°C | FAIL |
| 650FF | -11.33°C | -23.00°C | 40.67°C | 63.20-73.50°C | FAIL |
| 900FF | -2.75°C | -6.40°C | 41.12°C | 41.80-46.40°C | **WARN** |
| 950FF | -8.38°C | -20.20°C | 34.31°C | 35.50-38.50°C | FAIL |

## Investigation Performed

### Part A: Solar Gain Reduction for FF Cases

**Hypothesis**: FF min temps too warm because solar gains are overestimated. FF cases have no HVAC to offset gains, so solar directly heats the zone.

**Test**: Applied -15% to -25% solar gain reduction for FF cases

**Results**:
- 600FF: Min -10.02°C (worse, was -9.99°C) - WRONG DIRECTION
- 650FF: Min -11.34°C (essentially unchanged) - NO IMPROVEMENT
- 900FF: Min -3.08°C (worse, was -2.75°C) - REGRESSED TO FAIL
- 950FF: Min -8.51°C (slightly worse) - NO IMPROVEMENT
- Max temps all dropped significantly (38-39°C vs 40-41°C)

**Conclusion**: Reducing solar gains makes min temps warmer (less heat to lose at night). The approach was REVERTED.

### Part B: Internal Gains Verification

**Findings**:
- FF cases correctly defined with NO internal loads (per ASHRAE 140 spec)
- Model uses `spec.internal_loads` which is empty for FF cases
- `model.loads` vector is correctly initialized to 0.0 for FF cases
- Internal gains are NOT causing the temperature errors

**Conclusion**: Internal gains are correctly set to zero. No fix needed.

### Part C: Infiltration Review

**Findings**:
- Infiltration rate is 0.5 ACH for all FF cases (same as HVAC cases)
- Case 195 correctly has 0.0 ACH
- FF cases might need different infiltration rates per ASHRAE 140 spec

## Key Insight

The problem is structural - the 5R1C model has limitations for free-floating temperature prediction:
- 600/650 series (low-mass): Thermal capacitance too low to properly simulate temperature damping
- 900 series (high-mass): Better but still not matching reference swing

## Session 17/18 h_tr_em Multipliers (Already Tried)

| Case | h_tr_em_ff_multiplier | Result |
|------|---------------------|--------|
| 600FF | 6.5 | FAIL (min -9.99°C vs -18.80°C target) |
| 650FF | 6.5 | FAIL (min -11.33°C vs -23.00°C target) |
| 900FF | 2.8 | WARN (min -2.75°C vs -6.40°C target) |
| 950FF | 4.0 | FAIL (min -8.38°C vs -20.20°C target) |

## Results After Session 19 Investigation

| Metric | Value |
|--------|-------|
| FF Cases Improved | 0 |
| FF Cases Degraded | 1 (900FF: WARN→FAIL) |
| Regressions in annual energy | None |

**No improvements achieved. Solar gain reduction approach was tested and rejected.**

## Recommendations for Future Sessions

1. **Low-mass FF cases (600FF, 650FF)**: The 5R1C model may be fundamentally limited for these cases. Consider:
   - Testing 6R2C model (if not already used)
   - Adjusting thermal capacitance separately for FF cases
   - Investigating ASHRAE 140 reference model assumptions for low-mass FF

2. **High-mass FF cases (900FF, 950FF)**:
   - 900FF is already at WARN status - protect this!
   - 950FF may need different handling due to night ventilation interaction

3. **Structural limitation**: The 5R1C single-capacitance model may not capture the thermal response needed for accurate free-floating temperature prediction. Consider exploring 6R2C or 8R3C models.

4. **Alternative approach**: Rather than parameter tuning, investigate:
   - Weather data accuracy (solar radiation values)
   - Window optical properties (solar heat gain coefficient)
   - Surface heat transfer coefficients

## Session 19 Status: ❌ NO IMPROVEMENT

The investigation was completed but no improvements were achieved. Solar gain reduction was tested and found to make min temps warmer (opposite of desired). Internal gains were verified as correctly zero. The fundamental model limitation remains.
