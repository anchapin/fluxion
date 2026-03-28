# Physics-Based Refactoring - Session 20 Prompt

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Session 19 Recap
- **Solar gain reduction**: Tested -15% to -25% solar gain adjustment for FF cases
- **Result**: REVERSED - Made min temps WORSE (less solar = less heat to lose at night = warmer overnight temps)
- **Internal gains**: Verified as correctly set to zero for FF cases - not the issue
- **Key finding**: 5R1C model appears structurally limited for free-floating temperature prediction
- **Status**: 3/4 FF cases FAIL, 900FF WARN (unchanged from Session 17)
- **Pass rate**: 3.1% (unchanged)

---

## Session 20 Task: Explore Alternative Model Structures & Weather Data

### Objective
Since Sessions 17-19 have pushed parameter tuning to its limits and found the 5R1C model structurally limited for FF cases, this session explores alternative approaches:
1. Check if 6R2C model is being used for any FF cases (different thermal mass structure)
2. Investigate thermal capacitance adjustments specific to FF cases
3. Verify weather data accuracy (solar radiation values)
4. Explore infiltration rate adjustments for FF cases

### Background

**Session 17-19 Results Summary**:
- 600FF: Min -9.99°C (target: -18.80°C) - FAIL
- 650FF: Min -11.33°C (target: -23.00°C) - FAIL  
- 900FF: Min -2.75°C (target: -6.40°C) - **WARN** ✅
- 950FF: Min -8.38°C (target: -20.20°C) - FAIL

**Key Insight from Session 19**:
- Reducing solar gains makes min temps WORSE (counter-intuitive but verified)
- h_tr_em multipliers at local optimum (Session 18)
- 5R1C single-capacitance model may not capture FF thermal response

### Steps

#### Part A: Check 6R2C Model Usage for FF Cases (Priority 1)

1. **Find model selection logic**:
   - Search for "6R2C" or "six_r2c" in engine.rs
   - Check how model type is selected (5R1C vs 6R2C)
   - Determine if FF cases use 6R2C or 5R1C

2. **Test 6R2C for FF if not already used**:
   - If FF cases use 5R1C, try enabling 6R2C
   - 6R2C has separate envelope and internal thermal mass
   - May better capture free-floating thermal dynamics

3. **Validate**:
   - Run validation, check FF min/max temps
   - Ensure no regression in HVAC cases

#### Part B: Investigate Thermal Capacitance for FF Cases (Priority 2)

1. **Find thermal capacitance calculation**:
   - Search for "thermal_capacitance" or "Cm" in engine.rs
   - Check if FF cases have different capacitance than HVAC cases

2. **Test capacitance adjustment**:
   - Increase thermal capacitance for FF cases
   - More thermal mass = more temperature damping = different swing
   - Try +50% to +100% capacitance increase

3. **Validate**:
   - Run validation, check FF temps
   - Compare with reference values

#### Part C: Verify Weather Data (Priority 3)

1. **Find weather data source**:
   - Search for weather data loading in engine.rs
   - Check solar radiation values for Denver (ASHRAE location)

2. **Compare with reference**:
   - Verify hourly solar radiation values match ASHRAE 140 weather data
   - Check if any systematic offset exists

3. **Document findings**:
   - Note any discrepancies found

#### Part D: Explore Infiltration Rate for FF Cases (Priority 4)

1. **Check current infiltration**:
   - FF cases currently have 0.5 ACH (same as HVAC cases)
   - Per ASHRAE 140, FF cases might need different values

2. **Test infiltration adjustment**:
   - Try higher infiltration for FF cases (more heat loss at night = colder)
   - Try 1.0 ACH or 1.5 ACH

3. **Validate**:
   - Run validation, check min temps improve

### Expected Results After Fix

```
Free-Floating: Alternative approaches exploration
600FF: Min ~-12°C or better (was -9.99°C)
650FF: Min ~-12°C or better (was -11.33°C)  
900FF: Min ~-3°C (maintain WARN)
950FF: Min ~-10°C or better (was -8.38°C)
```

### Deliverable
- Summary of alternative approaches tested
- Implementation of successful approach (if any)
- Updated pass rate
- No regressions in annual energy

### Success Criteria
- [ ] At least one more FF case shows improvement
- [ ] 900FF maintains WARN status (no regression)
- [ ] No regressions in annual energy (600-series, 900-series)
- [ ] Document findings for future sessions

### Important Notes
- Don't break the annual energy validation from Sessions 17-19
- Free-floating temps are challenging - may need iterative approach
- Focus on physics-based parameters, NOT empirical corrections
- Run full validation after each significant change
- Use RUST_MIN_STACK=16777216 for release builds to avoid stack overflow