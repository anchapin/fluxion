# Physics-Based Refactoring - Session 18 Prompt

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Session 17 Recap
- **Free-floating temps**: Case-specific h_tr_em multipliers implemented (6.5x for low-mass, 2.8x for high-mass, 4.0x for 950FF)
- **Results**: 900FF maintains WARN status (-2.75°C within reference), 600FF/650FF improved but still FAIL
- **No regressions**: Annual energy values unchanged
- **Tests**: ASHRAE 140 validation tests pass (3/3)

---

## Session 18 Task: Try h_ve (Ventilation) Adjustment for FF Cases

### Objective
Continue improving free-floating temperature predictions by adjusting h_ve (ventilation heat transfer) as an alternative approach to h_tr_em.

### Background

**Session 17 Result**:
- 600FF: Min temp -9.99°C (target: -18.80°C) - still FAIL
- 650FF: Min temp -11.33°C (target: -23.00°C) - still FAIL
- 900FF: Min temp -2.75°C (target: -6.40°C) - **WARN** ✅
- 950FF: Min temp -8.38°C (target: -20.20°C) - still FAIL

**Hypothesis**: 
- h_tr_em alone not enough for low-mass cases
- Ventilation (h_ve) affects winter more than summer (larger temp difference)
- May need different approach for low-mass vs high-mass FF cases

### Steps

#### Part A: Increase h_ve for Low-Mass FF Cases (Priority 1)

1. **Find h_ve calculation in engine.rs**:
   - Search for "h_ve" in `src/sim/engine.rs`
   - Locate ventilation conductance calculation (around line ~1300-1340)

2. **Add h_ve multiplier for low-mass FF cases**:
   - Add similar match statement for h_ve_ff_multiplier
   - Try 2.0-3.0x increase for 600FF/650FF
   - Keep 1.0x for other cases (no HVAC impact)

3. **Run validation and check results**:
   - See if 600FF/650FF min temps improve
   - Check 900FF doesn't regress

#### Part B: Alternative - Adjust Thermal Capacitance (Priority 2)

If h_ve doesn't help enough:

1. **Find thermal capacitance (Cm) calculation**:
   - Search for "thermal_cap" or "kappa" in engine.rs
   - Around lines ~1336-1342

2. **Try reducing Cm for low-mass FF cases**:
   - Lower thermal mass = faster temperature swings
   - Try 0.5x reduction for 600FF/650FF
   - Keep 1.0x for other cases

#### Part C: Verify No Regressions (Priority 3)

1. **Run validation**:
   - Ensure 600-series annual energy still passes
   - Ensure 900-series annual energy still passes
   - Verify 900FF still WARN

### Expected Results After Fix

```
Free-Floating: Further improvement expected
600FF: Min ~-12°C or better (was -9.99°C)
650FF: Min ~-13°C or better (was -11.33°C)
900FF: Min ~-3°C (maintain WARN)
950FF: Min ~-10°C or better (was -8.38°C)
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
- [ ] Document findings for future sessions

### Important Notes
- Don't break the annual energy validation from Sessions 16-17
- Free-floating temps are challenging - may need iterative approach
- Focus on physics-based parameters, NOT empirical corrections
- Run full validation after each significant change
- Document what values work and what don't for future reference
- Use RUST_MIN_STACK=16777216 for release builds to avoid stack overflow