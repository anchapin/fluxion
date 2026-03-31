# Session 54: Free-Floating Temperature Validation

**Date**: 2026-03-27
**Follows**: Session 53 (Multi-Method Solver Manager)
**Status**: 📋 PLANNED
**Priority**: 🟡 MEDIUM - Fix free-floating temperature discrepancies
**Estimated Duration**: 1 week
**Prerequisite**: Session 53 successful (auto-selection working)

## Objective

Fix free-floating temperature discrepancies where max temps are 20-30°C below reference for 600-series and within range for 900-series. Free-floating cases test thermal dynamics without HVAC control.

## Context

### Current Free-Floating Results
| Case | Max Temp | Reference | Difference |
|------|----------|-----------|------------|
| 600FF | 45.66°C | 64.90-75.10°C | **20-30°C below** |
| 650FF | 43.71°C | 63.20-73.50°C | **20-30°C below** |
| 900FF | 47.94°C | 41.80-46.40°C | **1.5°C above** |
| 950FF | 37.67°C | 35.50-38.50°C | ✅ **Within range** |

### Problem
600-series max temps way too low - thermal mass not releasing heat correctly. 900-series close to reference.

## Investigation Plan

### Phase 1: Root Cause Analysis (Days 1-2)
- Compare 5R1C vs CTF vs FD for free-floating
- Check solar gain distribution (air vs mass)
- Verify thermal mass coupling (h_tr_em, h_tr_ms)
- Test with different solar fractions

### Phase 2: Solar Gain Redistribution (Days 2-4)
- Current: 70% to mass, 30% to air (for high-mass)
- Test: 50% to mass, 50% to air for low-mass
- Test: 30% to mass, 70% to air for low-mass
- Validate impact on max temps

### Phase 3: Thermal Mass Coupling (Days 4-5)
- Verify h_tr_em and h_tr_ms values
- Test different coupling ratios
- Check if coupling depends on HVAC mode
- Adjust for free-floating vs conditioned

### Phase 4: Validation (Days 5-6)
- Run all free-floating cases
- Compare with reference
- Verify within 5°C of reference

## Success Criteria
- [ ] Max temps within 5°C of reference
- [ ] Min temps within 5°C of reference
- [ ] ≥50% of free-floating cases passing
- [ ] No regressions in conditioned cases

## Expected Outcomes
- **Best Case**: All free-floating within 5°C
- **Medium Case**: 600-series improved but still 10°C below
- **Worst Case**: No improvement, accept as limitation

## Files to Modify
- `src/sim/engine.rs` (solar distribution, coupling)
- `src/validation/ashrae_140_cases.rs` (free-floating specs)

## Next Session
**Session 55**: Special Cases (960, 195)

## References
- `docs/ASHRAE140_ROADMAP.md` Phase 5
- `SESSION_45_SUMMARY.md` (free-floating analysis)

---

**Session 54 Goal**: Fix free-floating temperature discrepancies, achieving temps within 5°C of reference for ≥50% of free-floating cases.
