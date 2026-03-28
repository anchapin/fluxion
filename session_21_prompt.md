# Physics-Based Refactoring - Session 21 Prompt

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Session 20 Recap
- **Infiltration test**: 0.5→1.0 ACH had NO effect on FF temperatures
- **Thermal capacitance**: 0.25x, 0.5x, 2.0x tested - all made min temps WARMER
- **Key finding**: Thermal mass paradox - less mass makes temps warmer, more mass also makes them warmer
- **Root cause**: 5R1C single-capacitance model structure fundamentally limited for FF prediction
- **Status**: 3/4 FF cases FAIL, 900FF WARN (unchanged)
- **Pass rate**: 7.8% (unchanged from Session 19)

---

## Session 21 Task: Implement 6R2C Model for Free-Floating Cases

### Objective
Since Sessions 17-20 have found the 5R1C model structurally limited for FF cases, this session attempts to implement the 6R2C (two-capacitance) model specifically for FF cases to see if it better captures free-floating thermal dynamics.

### Background

**Session 20 Key Insight - Thermal Mass Paradox**:
- Less thermal mass → Faster response but WARMER night temps (less heat stored to release)
- More thermal mass → Slower response but WARMER night temps (heat releases slower)
- This suggests the single-capacitance model cannot capture the envelope/internal mass separation

**6R2C Model Advantages**:
- Two thermal mass nodes: envelope (walls, roof) + internal (furniture, partitions)
- Different time constants: envelope responds slower, internal responds faster
- May better capture diurnal temperature swings in free-floating conditions

### Steps

#### Part A: Enable 6R2C for FF Cases (Priority 1)

1. **Find 6R2C enable method**:
   - Search for `enable_6r2c` or similar in engine.rs
   - Check how to enable 6R2C for specific cases

2. **Test 6R2C for FF cases**:
   - Enable 6R2C mode when case_id ends with "FF"
   - Use default 6R2C parameters initially

3. **Validate**:
   - Run validation, check FF min/max temps
   - Ensure no regression in HVAC cases

#### Part B: Tune 6R2C Parameters if Needed (Priority 2)

If Part A shows improvement but not full:

1. **Adjust envelope/internal mass ratio**:
   - Try different splits between envelope and internal capacitance
   - Default might not be optimal for free-floating

2. **Adjust coupling conductance**:
   - h_tr_mi (mass-to-internal) conductance affects heat transfer rate
   - Try range of values

### Expected Results After Fix

```
Free-Floating: 6R2C model implementation
600FF: Min ~-15°C or better (was -10.42°C)
650FF: Min ~-18°C or better (was -11.55°C)  
900FF: Min ~-5°C (maintain WARN)
950FF: Min ~-15°C or better (was -8.87°C)
```

### Deliverable
- Summary of 6R2C implementation for FF cases
- Implementation of successful approach (if any)
- Updated pass rate
- No regressions in annual energy

### Success Criteria
- [ ] At least one more FF case shows improvement
- [ ] 900FF maintains WARN status (no regression)
- [ ] No regressions in annual energy (600-series, 900-series)
- [ ] Document findings for future sessions

### Important Notes
- Don't break the annual energy validation from Sessions 17-20
- Free-floating temps are challenging - may need iterative approach
- Focus on physics-based parameters, NOT empirical corrections
- Run full validation after each significant change
- Use RUST_MIN_STACK=16777216 for release builds to avoid stack overflow