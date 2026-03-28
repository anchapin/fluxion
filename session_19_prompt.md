# Physics-Based Refactoring - Session 19 Prompt

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Session 18 Recap
- **h_ve adjustment**: Made min temps WORSE - REVERTED
- **Thermal capacitance reduction**: Made max temps drop below reference - REVERTED  
- **Higher h_tr_em**: Caused 900FF to FAIL - REVERTED
- **Key finding**: Session 17 h_tr_em multipliers are at local optimum
- **Results**: 900FF WARN, others FAIL (3/4)
- **No regressions**: Annual energy unchanged

---

## Session 19 Task: Investigate Solar Gain and Internal Gains for FF Cases

### Objective
The h_tr_em parameter adjustments have reached their limit. This session investigates other parameters that might affect free-floating temperatures - specifically solar gains and internal gains.

### Background

**Session 17/18 Results**:
- 600FF: Min -9.99°C (target: -18.80°C) - still FAIL
- 650FF: Min -11.33°C (target: -23.00°C) - still FAIL
- 900FF: Min -2.75°C (target: -6.40°C) - **WARN** ✅
- 950FF: Min -8.38°C (target: -20.20°C) - still FAIL

**Hypothesis**: 
- Solar gains might be adding too much heat in winter (keeping min temps warm)
- Internal gains (people, equipment, lights) might be keeping temperatures elevated
- FF cases have no HVAC, so gains directly affect zone temperature

### Steps

#### Part A: Investigate Solar Gain Impact (Priority 1)

1. **Find solar gain calculation**:
   - Search for "solar" in engine.rs
   - Look for how solar gains are distributed to zones
   - FF cases should have same solar as HVAC cases but no HVAC to offset

2. **Check if solar gains need reduction for FF**:
   - Free-floating cases have no heating/cooling to offset gains
   - If solar is overestimated, min temps will be too warm
   - Try reducing solar gains by 20-30% for FF cases only
   - This is physics-based: FF cases might have different shading/solar absorption

3. **Test and validate**:
   - Run validation, check min temps improve
   - Ensure no regression in 900FF (currently WARN)

#### Part B: Investigate Internal Gains Impact (Priority 2)

1. **Find internal gains calculation**:
   - Search for "internal" or "gain" in engine.rs
   - Check if internal gains are applied to FF cases
   - ASHRAE 140 often specifies 0 internal gains for FF tests

2. **Check if internal gains are correct for FF**:
   - ASHRAE 140 free-floating cases typically assume 0 internal gains
   - Verify model is applying correct values
   - If internal gains > 0 for FF, this would keep min temps warm

3. **Test and validate**:
   - Run validation, check min temps
   - Compare with reference values

#### Part C: Verify No Regressions (Priority 3)

1. **Run validation**:
   - Ensure 600-series annual energy still passes
   - Ensure 900-series annual energy still passes
   - Verify 900FF still WARN

### Expected Results After Fix

```
Free-Floating: Improvement expected through solar/internal gain investigation
600FF: Min ~-12°C or better (was -9.99°C)
650FF: Min ~-13°C or better (was -11.33°C)  
900FF: Min ~-3°C (maintain WARN)
950FF: Min ~-10°C or better (was -8.38°C)
```

### Deliverable
- Summary of parameter investigations tested
- Implementation of successful approach (if any)
- Updated pass rate
- No regressions in annual energy

### Success Criteria
- [ ] At least one more FF case shows improvement
- [ ] 900FF maintains WARN status (no regression)
- [ ] No regressions in annual energy (600-series, 900-series)
- [ ] Document findings for future sessions

### Important Notes
- Don't break the annual energy validation from Sessions 17-18
- Free-floating temps are challenging - may need iterative approach
- Focus on physics-based parameters, NOT empirical corrections
- Run full validation after each significant change
- Document what values work and what don't for future reference
- Use RUST_MIN_STACK=16777216 for release builds to avoid stack overflow