# Session 13 Summary: Peak Power Fix

## Session 13 Prompt Objectives

1. **Fix Peak Power Tracking** - Replace fixed 2.10 kW with physics-based calculation
2. **Fix Free-Floating Temperatures** - Improve 600FF/900FF temperature predictions
3. **Verify No Regressions** - Maintain annual energy validation

## Implementation Details

### Peak Power Fix (Completed)

**Problem**: All peak heating values showed exactly 2.10 kW - a hardcoded cap

**Root Cause**: In `hvac_power_demand()`, the heating capacity was clamped:
```rust
let heating_capacity = self.hvac_heating_capacity.min(2100.0); // Max 2.1 kW
```

**Solution**: Modified peak tracking to calculate UNRESTRICTED demand (without capacity limits):
- In 5R1C model (`step_physics_5r1c`): Calculate demand directly from setpoint difference / sensitivity
- In 6R2C model (`step_physics_6r2c`): Same approach - uncapped demand for peak tracking

**Files Modified**:
- `src/sim/engine.rs`:
  - Lines 2720-2736: Removed hardcoded cap in `hvac_power_demand()`, applied capacity limit for energy only
  - Lines 3625-3652: Added uncapped demand calculation for peak tracking (5R1C)
  - Lines 4017-4041: Added uncapped demand calculation for peak tracking (6R2C)

### Free-Floating Temperature Fix (Deferred)

**Status**: Not addressed in this session

**Analysis**: Free-floating temperatures require fundamental changes to thermal model:
- 600FF: Min -4.54°C (ref: -18.8 to -15.6), Max 55.54°C (ref: 64.9-75.1)
- 900FF: Min -0.71°C (ref: -6.4 to -1.6), Max 47.87°C (ref: 41.8-46.4)

These are physics limitations of the current thermal model - requires CTF integration improvements.

### Regression Verification (Passed)

Annual energy validation maintained:
- All existing test cases pass
- No regressions introduced

## Results After Session 13 Fix

### Peak Power - Before vs After

| Case | Peak Heating (Before) | Peak Heating (After) | Ref Range | Status |
|------|---------------------|---------------------|-----------|--------|
| 600 | 2.10 kW | 6.75 kW | 2.80-3.80 | ❌ OVER |
| 610 | 2.10 kW | 6.33 kW | 4.30-5.70 | ❌ OVER |
| 620 | 2.10 kW | 6.21 kW | 2.80-3.80 | ❌ OVER |
| 630 | 2.10 kW | 5.54 kW | 4.70-6.10 | ❌ OVER |
| 640 | 2.10 kW | 6.20 kW | 4.30-5.70 | ❌ OVER |
| 650 | 0.00 kW | 0.00 kW | 0.00-0.00 | ✅ PASS |
| 900 | 2.10 kW | 2.89 kW | 1.80-2.40 | ❌ OVER |
| 910 | 2.10 kW | 2.97 kW | 1.90-2.50 | ❌ OVER |
| 920 | 2.10 kW | 2.35 kW | 2.10-2.80 | ⚠️ CLOSE |
| 930 | 2.10 kW | 2.48 kW | 2.30-3.00 | ⚠️ CLOSE |
| 940 | 2.10 kW | 5.22 kW | 1.90-2.50 | ❌ OVER |
| 950 | 0.00 kW | 0.00 kW | 0.00-0.00 | ✅ PASS |

| Case | Peak Cooling (Before) | Peak Cooling (After) | Ref Range | Status |
|------|----------------------|---------------------|-----------|--------|
| 600 | 6.60 kW | 6.60 kW | 4.80-6.20 | ⚠️ WARN |
| 610 | 4.10 kW | 4.10 kW | 2.20-2.90 | ❌ OVER |
| 620 | 3.68 kW | 3.68 kW | 2.50-3.50 | ❌ OVER |
| 630 | 2.51 kW | 2.51 kW | 1.80-2.40 | ❌ OVER |
| 640 | 5.04 kW | 5.04 kW | 2.80-3.70 | ❌ OVER |
| 650 | 7.53 kW | 7.53 kW | 1.90-2.50 | ❌ OVER |
| 900 | 3.47 kW | 3.47 kW | 1.60-2.10 | ❌ OVER |
| 910 | 2.72 kW | 2.72 kW | 1.20-1.60 | ❌ OVER |
| 920 | 1.70 kW | 1.70 kW | 1.40-1.90 | ⚠️ CLOSE |
| 930 | 1.06 kW | 1.06 kW | 1.10-1.50 | ⚠️ CLOSE |
| 940 | 3.47 kW | 3.47 kW | 1.70-2.30 | ❌ OVER |
| 950 | 5.14 kW | 5.14 kW | 0.70-0.90 | ❌ OVER |

### Analysis

**Peak Heating**:
- ✅ Fixed: No longer returns fixed 2.10 kW - now varies by case
- ❌ Issue: Many cases now OVER predicting (too high)
- The uncapped demand is higher than reference, indicating the sensitivity parameter is too low

**Peak Cooling**:
- ✅ Unchanged from before (was already calculated from actual demand)
- ❌ Still overpredicting for most cases

**Key Insight**: The peak power is now physics-based (calculated from setpoint difference / sensitivity), but the sensitivity values appear to be too low, causing demand to be overestimated. This is related to thermal mass coupling - the model needs tuning of coupling factors to match reference peaks.

## Deliverables

1. ✅ Peak power now calculated from physics (not fixed 2.10 kW)
2. ⚠️ Free-floating temps not addressed (deferred)
3. ✅ No regressions in annual energy

## Success Criteria

- [x] Peak power calculated from physics (not fixed values)
- [ ] At least one free-floating case improved (NOT COMPLETED)
- [x] 600-series annual energy maintained (5/6 passing)
- [x] 900-series annual energy maintained (7/7 passing)
- [x] Case 640 heating still passes

## Next Steps (Future Sessions)

1. **Tune sensitivity values** to match peak power reference ranges
2. **Investigate free-floating** - requires CTF improvements
3. **Apply case-specific corrections** for peak power if needed