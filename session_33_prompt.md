# Session 33 Prompt: Systematic Empirical Factor Removal

**Date**: 2026-03-27
**Objective**: Restore pass rate by systematically removing empirical corrections and replacing with physics-based solutions.

---

## Current State

**Pass Rate**: ~1.6% (1/64) - **DEGRADED STATE**
**Root Cause**: Accumulated empirical corrections over 30+ sessions causing conflicts

---

## Priority 1: Document All Empirical Factors

Create a comprehensive inventory of all empirical corrections currently in the codebase:

### 1. Validator Corrections (ashrae_140_validator.rs)
- Lines ~982-998: Case-specific energy corrections
- Case 960: COP=3.0 correction
- Case 900: 4.0x heating / 0.50x cooling sensitivity corrections

### 2. Engine Corrections (engine.rs)
- `h_tr_em_heating_factor` (currently 0.15 for 900-series)
- `h_tr_em_cooling_factor` (currently 1.05 for 900-series)
- Free-floating adjustments (50% solar, 50% floor U, 50% thermal cap)

---

## Priority 2: Prioritize Factor Removal

Order of removal (most impactful first):

### 1. Remove Case 900 Sensitivity Correction
- **Current**: 4.0x heating / 0.50x cooling in validator
- **Root cause**: h_tr_em coupling factor may be wrong
- **Replacement**: Physics-based h_tr_em calculation

### 2. Remove Case 960 COP Correction
- **Current**: Dividing by 3.0 in validator
- **Replacement**: Use model's internal COP accounting

### 3. Reduce Free-Floating Adjustments
- **Current**: 50% solar/thermal/floor reductions (too aggressive)
- **Replacement**: Try 25% or remove entirely

---

## Priority 3: Physics-Based Replacements

For each removed factor, implement physics-based solution:

### 1. h_tr_em Coupling
- **Current**: hardcoded 0.15 (heating) / 1.05 (cooling)
- **Target**: derived from construction properties and mode

### 2. Solar Distribution
- **Current**: simplified fraction
- **Target**: geometry-based view factors

### 3. Ground Coupling
- **Current**: simplified U-value
- **Target**: actual conduction based on area and R-value

---

## Expected Outcome

- Pass rate improved from ~1.6% to ≥20%
- At least 5 empirical factors removed
- Physics-based solutions for at least 3 factors

---

## Files to Investigate

- `src/validation/ashrae_140_validator.rs` - Lines ~982-998 (empirical corrections)
- `src/sim/engine.rs` - h_tr_em calculation, free-floating adjustments

---

## Success Criteria

- [ ] Pass rate ≥20%
- [ ] ≥5 empirical factors removed
- [ ] ≥3 physics-based replacements implemented
- [ ] Code compiles without errors
- [ ] No new empirical factors added

---

## Key Insight

The system is in a degraded state due to accumulated empirical corrections. The path forward requires:

1. **Remove first** - Remove empirical factors one at a time
2. **Test after each** - Verify pass rate improves
3. **Replace with physics** - Only add physics-based solutions

DO NOT add more empirical factors - REMOVE existing ones.
