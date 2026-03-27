# Session 34 Prompt: Fix Fundamental Thermal Model Physics

**Date**: 2026-03-27
**Objective**: Continue removing empirical factors and fix fundamental physics issues to improve pass rate from current baseline.

---

## Current State

**Pass Rate**: 1.6% (1/64) - **BASELINE REVEALED**

After Session 33 removed 9 empirical factors, the model now produces:
- 2-3x higher energy than ASHRAE 140 reference
- This reveals the root cause is in thermal model physics, not corrections

---

## Priority 1: Analyze Root Cause - Why Model Overpredicts 2-3x

The model produces ~2-3x higher energy than reference. Investigate:

### 1.1 Thermal Mass Coupling (h_tr_em)
- Current: (1.0, 1.0) - is this too high?
- Check if coupling ratio should be < 1.0 for high-mass buildings
- Compare with ISO 13790 or ASHRAE 140 formulae

### 1.2 Sensitivity Calculation
- Is HVAC sensitivity (W/K) correct?
- Check: sensitivity = h_tr_ms * h_tr_is / (h_tr_ms + h_tr_is)
- May need mode-specific adjustments

### 1.3 Solar Gains Distribution
- What fraction goes directly to zone air vs. thermal mass?
- Is view-factor calculation correct?

### 1.4 Ground Coupling
- Is floor heat loss too high?
- Check floor U-value calculation

---

## Priority 2: Fix 600-Series (Low-Mass) Cases

Currently failing with 8-9 MWh heating vs 5-7 MWh reference:

| Case | Current | Reference |
|------|---------|-----------|
| 600 | 8.65 MWh | 5.50-7.50 MWh |
| 610 | 9.08 MWh | 4.36-5.79 MWh |
| 620 | 7.90 MWh | 4.50-6.50 MWh |

Potential fixes:
- Check internal gains (currently 0 for 600-series?)
- Verify window/solar gains distribution
- Check HVAC sensitivity for low-mass buildings

---

## Priority 3: Fix Free-Floating Cases

| Case | Current Min | Ref Min | Current Max | Ref Max |
|------|-------------|----------|-------------|----------|
| 600FF | -6.70°C | -18.8°C | 38.88°C | 64.9°C |
| 900FF | -3.51°C | -6.4°C | 38.03°C | 41.8°C |

Current approach (50% solar, 50% floor U, 50% thermal cap) isn't working - needs physics-based approach.

---

## Priority 4: Address Remaining Empirical Factors

Check if any other empirical factors remain:
- Any remaining case-specific corrections in validator?
- Any hardcoded factors in engine?

---

## Expected Outcome
- Pass rate improved from 1.6% to ≥10%
- Root cause of overprediction identified
- At least one physics-based fix implemented

---

## Files to Investigate
- `src/sim/engine.rs` - h_tr_em calculation, sensitivity, solar distribution
- `src/physics/cta.rs` - VectorField operations
- `src/validation/ashrae_140_validator.rs` - Any remaining corrections

---

## Success Criteria
- [ ] Pass rate ≥10%
- [ ] Root cause of overprediction identified
- [ ] At least one physics-based fix implemented
- [ ] Code compiles without errors
- [ ] No new empirical factors added

---

## Key Insight

After removing empirical corrections, the model shows its true physics. The overprediction indicates the thermal model equations need fundamental fixes, not empirical patches. Focus on:

1. **h_tr_em calculation** - derive from thermal network physics
2. **Sensitivity calculation** - verify HVAC sensitivity formula
3. **Solar distribution** - use geometry-based view factors
4. **Ground coupling** - calculate from floor area and R-value

DO NOT add new empirical factors - fix the physics.
