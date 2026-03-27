# Session 33 Summary: Empirical Factor Removal - Baseline Established

**Date**: 2026-03-27
**Status**: COMPLETE - Baseline established for physics-based approach

---

## Objective

Systematically remove empirical corrections from the codebase to restore a physics-based thermal model.

---

## Changes Made

### 1. Validator Corrections Removed (ashrae_140_validator.rs)

**Case 960 COP/efficiency corrections (Lines 987-992)**:
- REMOVED: `cooling_cop = 3.0`, `heating_efficiency = 0.9`
- Original: Divided heating/cooling by these factors to match reference

**900-series sensitivity corrections (Lines 997-1019)**:
- REMOVED Case 900: 4.0x heating, 0.50x cooling
- REMOVED Case 910: 2.5x heating, 0.35x cooling
- REMOVED Case 940: 2.7x heating, 0.45x cooling
- REMOVED Case 950: 0.35x cooling

### 2. Engine Corrections Removed/Replaced (engine.rs)

**h_tr_em coupling factors (Lines 1115-1132)**:
- REPLACED: `(0.15, 1.05)` → `(1.0, 1.0)`
- Removed hardcoded empirical coupling factors for heating/cooling modes

**Sensitivity correction (Lines 1138-1144)**:
- REMOVED: Case 900 had 4.0x sensitivity correction
- All cases now use 1.0 (no correction)

---

## Results After Removal

| Case | Heating (MWh) | Ref Heating | Cooling (MWh) | Ref Cooling | Status |
|------|---------------|-------------|---------------|-------------|--------|
| 900 | 4.75 | 1.17-2.04 | 6.95 | 2.13-3.67 | ❌ |
| 910 | 5.23 | 1.51-2.28 | 4.83 | 0.82-1.88 | ❌ |
| 920 | 4.07 | 3.26-4.30 | 2.42 | 1.84-3.31 | ❌ |
| 930 | 5.26 | 4.14-5.34 | 1.04 | 1.04-2.24 | ❌ |
| 940 | 4.14 | 0.79-1.41 | 6.95 | 2.08-3.55 | ❌ |
| 950 | 0.00 | 0.00-0.00 | 2.73 | 0.39-0.92 | ❌ |
| 960 | 0.91 | 5.00-15.00 | 4.22 | 1.00-3.50 | ❌ |

### Key Findings

1. **Model produces ~2-3x higher heating energy than reference**: 4-5 MWh vs 1-2 MWh
2. **Model produces ~2-3x higher cooling energy than reference**: 3-7 MWh vs 1-4 MWh
3. **No empirical corrections = baseline physics revealed**
4. **Root cause not in empirical factors - in thermal model physics itself**

---

## Root Cause Analysis

The overprediction stems from fundamental thermal model issues:

1. **Thermal mass coupling**: h_tr_em too high → too much heat flows to/from thermal mass
2. **Sensitivity**: Model sensitivity too low → HVAC stays on longer
3. **Solar distribution**: May be incorrect distribution of solar gains
4. **Ground coupling**: Floor coupling may be too strong

---

## Next Steps (Session 34+)

1. **Analyze thermal model equations** - Check h_tr_em calculation
2. **Review solar gain distribution** - May need view factor-based approach
3. **Check thermal capacitance** - May be overestimating thermal mass
4. **Verify ground coupling** - Floor U-value may be too high

---

## Files Modified

- `src/validation/ashrae_140_validator.rs` - Removed 5 empirical correction blocks
- `src/sim/engine.rs` - Removed/replaced h_tr_em coupling factors and sensitivity correction

---

## Success Criteria Assessment

| Criterion | Target | Result | Status |
|-----------|--------|--------|--------|
| Pass rate ≥20% | ≥20% | 1.6% | ❌ |
| ≥5 empirical factors removed | ≥5 | 9 factors removed | ✅ |
| ≥3 physics-based replacements | ≥3 | 0 (baseline only) | ❌ |
| Code compiles | Yes | Yes | ✅ |
| No new factors added | 0 | 0 | ✅ |

**Overall**: Partial success - factors removed, baseline revealed, but physics still needs work.
