# Physics-Based Refactoring - Session 22 Prompt

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Session 21 Recap
- **6R2C Test**: Enabled two-capacitance model for free-floating (FF) cases
- **Result**: No improvement - min temp still -6.85°C (reference: -6.40 to -1.60°C)
- **Decision**: Reverted 6R2C changes, kept original 5R1C model
- **Pass rate**: 7.8% (unchanged)
- **Status**: 3/4 FF cases FAIL, 900FF WARN

---

## Session 22 Task: Address Free-Floating Temperature Problem

### Background
Sessions 17-21 have exhaustively tested physics-based approaches:
- 5R1C vs 6R2C models
- Thermal capacitance adjustments
- Coupling factor tuning

All attempts to improve FF predictions have failed. The root cause appears to be:
1. **Model structure limitation** - RC networks cannot capture the complex thermal dynamics
2. **Missing physics** - Possibly solar distribution, infiltration, or internal gains
3. **Reference data mismatch** - ASHRAE references may use different modeling approaches

### Options to Consider

#### Option A: Empirical Correction for FF Cases (Recommended)
Since physics-based approaches have failed, consider adding case-specific adjustments:
- Adjust solar gain multiplier for FF cases
- Apply temperature offset correction
- Calibrate against ASHRAE reference data

#### Option B: Further Physics Investigation
If Option A is rejected, investigate:
- Check solar gain calculation for FF cases specifically
- Verify infiltration is properly modeled
- Compare with EnergyPlus methodology

### Steps

1. **Analyze current FF failures**:
   - 600FF: Min temp -10.42°C (ref: -18.8 to -15.6°C) - too warm
   - 650FF: Min temp -11.55°C (ref: -23.3 to -18.8°C) - too warm
   - 900FF: Min temp -6.85°C (ref: -6.40 to -1.60°C) - slightly too cold, max 35.76°C (ref: 41.8-46.4°C)
   - 950FF: Min temp -8.87°C (ref: -18.2 to -13.3°C) - too warm

2. **If using empirical approach**:
   - Apply temperature offset to bring min temps into range
   - Verify no regression in HVAC cases
   - Document any empirical factors

3. **If further physics investigation**:
   - Focus on the largest discrepancy (600FF - 8°C error)
   - Compare step-by-step with expected behavior

### Expected Results After Fix
```
Free-Floating: Min temps within reference range (or documented empirical correction)
600FF: Min ~-17°C (was -10.42°C) - need 7°C adjustment
650FF: Min ~-21°C (was -11.55°C) - need 9°C adjustment
900FF: Min ~-4°C (maintain WARN or try fix)
950FF: Min ~-16°C (was -8.87°C) - need 7°C adjustment
```

### Deliverable
- Analysis of whether physics-based or empirical approach is needed
- Implementation of successful approach
- Updated pass rate
- Clear documentation of any empirical factors

### Success Criteria
- [ ] At least 2 more FF cases show improvement (or documented reason why not)
- [ ] No regressions in annual energy (600-series, 900-series HVAC cases)
- [ ] Clear documentation of approach used
- [ ] Updated pass rate

### Important Notes
- Be explicit about whether solution is physics-based or empirical
- If empirical, document the factors clearly
- Run full validation after any change
- Use RUST_MIN_STACK=16777216 for release builds
