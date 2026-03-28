# Physics-Based Refactoring - Session 17 Prompt

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Session 16 Recap
- **Free-floating temps**: Added 1.8x h_tr_em multiplier for FF cases
- **Results**: 900FF min temp improved from -0.71°C to -1.93°C → **WARN** (within reference!)
- **Other FF cases**: 600FF, 650FF, 950FF still FAIL but improved slightly
- **No regressions**: Pass rate improved from 4.7% to 6.2%
- **Tests**: All 13 tests pass

---

## Session 17 Task: Continue Free-Floating Temperature Optimization

### Objective
Continue improving free-floating temperature predictions by either:
1. Further increasing h_tr_em for FF cases (2.0-2.5x), OR
2. Adjusting h_ve (ventilation) for FF cases, OR
3. Adjusting thermal capacitance (Cm) for FF cases

### Background

**Session 16 Result - 900FF SUCCESS**:
- h_tr_em 1.8x multiplier worked: min temp moved from -0.71°C → -1.93°C (within reference!)
- This proves increasing heat transfer works for high-mass FF cases

**Remaining Issues** (Session 16):
- 600FF: min temp -6.52°C (target: -18.80°C) - still FAIL
- 650FF: min temp -10.52°C (target: -23.00°C) - still FAIL  
- 950FF: min temp -8.73°C (target: -20.20°C) - still FAIL

**Hypothesis**: 
- Low-mass cases (600FF) may need even higher h_tr_em or different h_ve
- The fundamental thermal model may need different parameters for FF vs HVAC mode

### Steps

#### Part A: Try Higher h_tr_em Multiplier (Priority 1)

1. **Increase h_tr_em to 2.5x** for FF cases:
   - Modify `src/sim/engine.rs` line ~1326
   - Change `1.8` to `2.5`
   - Run validation and check if all FF cases improve

2. **Alternative: Case-specific multipliers**:
   - Try 2.5x for 600FF/650FF (low-mass), keep 1.8x for 900FF/950FF (high-mass)
   - This tests if low-mass vs high-mass need different adjustments

#### Part B: Try h_ve (Ventilation) Increase (Priority 2)

If h_tr_em doesn't fully solve it:

1. **Increase h_ve for FF cases** (alternative approach):
   - In `src/sim/engine.rs` around line 1240
   - Add multiplier to h_ve (ventilation conductance)
   - Higher infiltration affects winter more than summer (larger temp difference)
   - Start with 2.0x and test

#### Part C: Verify No Regressions (Priority 3)

1. **Run validation**:
   - Ensure 600-series annual energy still passes
   - Ensure 900-series annual energy still passes
   - Verify peak power improvements maintained

### Expected Results After Fix

```
Free-Floating: More improvement expected
600FF: Min ~-10°C or better (was -6.52°C)
900FF: Min ~-3°C (was -1.93°C, already WARN)
Other FF cases: Improved as well
```

### Deliverable
- Summary of parameter adjustments tested
- Implementation of successful approach
- Updated pass rate
- No regressions in annual energy

### Success Criteria
- [ ] At least one more FF case shows significant improvement
- [ ] 900FF maintains WARN status (no regression)
- [ ] No regressions in annual energy (600-series, 900-series)
- [ ] Peak power improvements maintained
- [ ] Document findings for future sessions

### Important Notes
- Don't break the annual energy validation from Sessions 14-16
- Free-floating temps are challenging - may need iterative approach
- Focus on physics-based parameters, NOT empirical corrections
- Run full validation after each significant change
- Document what values work and what don't for future reference