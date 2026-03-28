# Session 55: Special Cases Validation

**Date**: 2026-03-27
**Follows**: Session 54 (Free-Floating Validation)
**Status**: 📋 PLANNED
**Priority**: 🟢 MEDIUM - Fix edge cases (multi-zone, steady-state)
**Estimated Duration**: 1 week
**Prerequisite**: Session 54 successful (free-floating improved)

## Objective

Validate and fix special cases (960 sunspace, 195 steady-state) which test edge cases of the solver - multi-zone heat transfer and steady-state conduction.

## Context

### Current Special Case Results
| Case | Description | Issue |
|------|-------------|-------|
| 960 | Sunspace (2-zone) | Inter-zone heat transfer |
| 195 | Steady-state (no windows) | Envelope conduction only |

### Case 960: Sunspace
- **Problem**: Multi-zone heat transfer not validated
- **Expected**: Sunspace warmer than main zone
- **Current**: Results not matching reference

### Case 195: Steady-State
- **Problem**: Solid conduction only
- **Expected**: No solar gains, steady envelope heat transfer
- **Current**: Annual heating 4.85 MWh (Ref: 3.50-6.00)

## Implementation Plan

### Phase 1: Case 960 Multi-Zone (Days 1-3)
- Verify inter-zone heat transfer calculation
- Check sunspace temperature modeling
- Validate solar gain distribution to sunspace
- Test common wall conductance

### Phase 2: Case 195 Steady-State (Days 3-4)
- Verify steady-state conduction
- Check envelope heat transfer
- Validate no solar gains condition
- Test thermal mass in steady-state

### Phase 3: Validation (Days 5-6)
- Run both cases
- Compare with reference
- Debug issues
- Document results

## Success Criteria
- [ ] Case 960 passing (within reference ranges)
- [ ] Case 195 passing (within reference ranges)
- [ ] Multi-zone solver validated
- [ ] Steady-state conduction validated

## Expected Outcomes
- **Best Case**: Both cases passing
- **Medium Case**: One passing, one close
- **Worst Case**: Both still failing, accept as limitations

## Files to Examine
- `src/sim/interzone.rs` (inter-zone heat transfer)
- `src/validation/ashrae_140_cases.rs` (case specs)

## Next Session
**Session 56**: Final Validation

## References
- `docs/ASHRAE140_ROADMAP.md` Phase 6

---

**Session 55 Goal**: Validate special cases (960, 195), achieving ≥50% pass rate for special cases and validating multi-zone and steady-state capabilities.
