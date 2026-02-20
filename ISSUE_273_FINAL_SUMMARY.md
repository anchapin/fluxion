# Issue #273: Multi-Zone Sunspace Investigation - Final Summary

## Overview

Investigated Case 960 multi-zone sunspace simulation where cooling energy was 20x higher than reference values.

## Root Causes Identified

### 1. HVAC Applied to All Zones (FIXED)

**Issue**: The `hvac_power_demand` method calculated HVAC for all zones without checking the `hvac_enabled` field.

**Impact**: Zone 1 (sunspace) was receiving HVAC control, generating massive energy consumption.

**Fix**:
- Set `hvac_enabled` field from `CaseSpec.hvac` array (Zone 0 enabled, Zone 1 disabled)
- Modified `hvac_power_demand` to multiply HVAC demand by `hvac_enabled` flags

**Result**: HVAC is now correctly disabled for free-floating zones.

### 2. Zone Areas Not Zone-Specific (FIXED)

**Issue**: All zones used the first zone's floor area (48 m²) instead of zone-specific areas.

**Impact**: Zone 1 (sunspace, 16 m²) had 3x the thermal capacitance it should have.

**Fix**: Set zone-specific floor areas from `spec.geometry`.

**Result**: Zone 0 has 48 m², Zone 1 has 16 m².

### 3. Thermal Parameters Not Zone-Specific (FIXED)

**Issue**: All thermal conductances and capacitances were calculated using the first zone's geometry and applied to all zones.

**Impact**:
- Window conductance: Both zones had same value (incorrect)
- Infiltration conductance: Both zones had same value (incorrect)
- Thermal capacitance: Both zones had same value (incorrect)
- Other conductances (h_tr_is, h_tr_ms, h_tr_em): Same for both zones (incorrect)

**Fix**: Implemented zone-specific calculation loop that calculates each thermal parameter per zone using zone-specific geometry.

**Result**: Zone-specific thermal parameters now correctly calculated:
- Zone 0 (48 m²): 19.9 MJ/K thermal capacitance, 36 W/K window conductance
- Zone 1 (16 m²): 10.4 MJ/K thermal capacitance, 18 W/K window conductance

### 4. Window Areas Not Defined for Zone 0 (FIXED)

**Issue**: Case 960 builder didn't add windows to Zone 0 (back-zone).

**Impact**: Back-zone had no windows (-0.00 m²), which is incorrect per ASHRAE 140 spec.

**Fix**: Added south-facing window (12 m²) to Zone 0 in Case 960 builder.

**Result**: Zone 0 has 12 m² south window, Zone 1 has 6 m² south window.

## Current Simulation Results

### Before All Fixes
```
=== ASHRAE 140 Case 960 Results (Before All Fixes) ===
Annual Heating: 75.45 MWh (reference: 1.65-2.45 MWh)  ← 30-46x too high
Annual Cooling: 0.15 MWh (reference: 1.55-2.78 MWh)   ← 10x too low
Peak Heating: 22.05 kW (reference: 2.20-2.90 kW)     ← ~7-10x too high
Peak Cooling: 0.88 kW (reference: 1.50-2.00 kW)      ← Too low
=== End ===
```

### After HVAC Enable Fix Only
```
=== ASHRAE 140 Case 960 Results (After HVAC Fix) ===
Annual Heating: 61.02 MWh (reference: 1.65-2.45 MWh)  ← ~25-37x too high
Annual Cooling: 0.10 MWh (reference: 1.55-2.78 MWh)   ← ~15x too low
Peak Heating: 17.09 kW (reference: 2.20-2.90 kW)     ← ~6-8x too high
Peak Cooling: 0.57 kW (reference: 1.50-2.00 kW)      ← Too low
=== End ===
```

### After All Fixes (Current State)
```
=== ASHRAE 140 Case 960 Results (After All Fixes) ===
Annual Heating: 49.08 MWh (reference: 1.65-2.45 MWh)  ← ~20-30x too high
Annual Cooling: 0.02 MWh (reference: 1.55-2.78 MWh)   ← ~70x too low
Peak Heating: 14.09 kW (reference: 2.20-2.90 kW)     ← ~5-6x too high
Peak Cooling: 0.28 kW (reference: 1.50-2.00 kW)      ← Too low
=== End ===
```

## Remaining Issues

### 1. Back-Zone Temperature is Perfectly Flat

**Observation**: Back-zone temperature remains exactly 20°C at all timesteps.

**Implication**: This suggests the HVAC is over-controlling the back-zone, keeping it exactly at the setpoint without allowing natural fluctuations.

**Possible Causes**:
- HVAC sensitivity calculation may be incorrect for multi-zone buildings
- Superposition calculation in temperature update may have issues
- Inter-zone heat transfer may not be properly accounted for

### 2. Cooling Energy is Too Low

**Observation**: Cooling energy is 0.02 MWh vs 1.55-2.78 MWh reference.

**Implication**: The sunspace is not transferring enough heat to the back-zone in summer.

**Possible Causes**:
- Inter-zone conductance (111.64 W/K) may be too low
- Solar gains distribution may be incorrect
- Sunspace may not be overheating enough due to incorrect thermal parameters

### 3. Peak Heating is Still Too High

**Observation**: Peak heating is 14.09 kW vs 2.20-2.90 kW reference.

**Implication**: The HVAC system is requesting excessive heating power.

**Possible Causes**:
- Sensitivity calculation may be incorrect
- Inter-zone heat transfer may be creating excessive load on back-zone
- Thermal parameters may still need calibration

## Files Modified

1. `/home/alexc/Projects/fluxion/src/sim/engine.rs`
   - Added zone-specific HVAC enable flag setting
   - Modified `hvac_power_demand` to respect `hvac_enabled`
   - Implemented zone-specific thermal parameter calculations
   - Fixed zone area setting for multi-zone

2. `/home/alexc/Projects/fluxion/src/validation/ashrae_140_cases.rs`
   - Added `window_area_by_zone_and_orientation()` method
   - Fixed Case 960 builder to add windows to Zone 0

3. Test files added:
   - `tests/test_issue_273_multi_zone_parameters.rs`
   - `tests/test_issue_273_window_debug.rs`

4. Documentation:
   - `docs/ISSUE_273_ROOT_CAUSE_ANALYSIS.md` (root cause analysis)

## Progress Summary

| Issue | Status | Impact |
|-------|--------|---------|
| HVAC enabled for all zones | FIXED | Heating reduced from 75 to 61 MWh (~19% reduction) |
| Zone areas not zone-specific | FIXED | Correct thermal capacitance per zone |
| Thermal parameters not zone-specific | FIXED | All conductances now zone-specific |
| Window areas missing for Zone 0 | FIXED | Zone 0 now has 12 m² south window |
| Back-zone temperature flat | INVESTIGATING | Temp stays exactly at 20°C (suspicious) |
| Cooling energy too low | INVESTIGATING | Need to check inter-zone heat transfer |
| Peak heating too high | INVESTIGATING | Sensitivity calculation may be incorrect |

## Next Steps

1. **Investigate Temperature Update Logic**
   - Check why back-zone temperature is perfectly flat
   - Review superposition calculation in `step_physics`
   - Verify inter-zone heat transfer is properly included

2. **Calibrate Inter-Zone Conductance**
   - Review 111.64 W/K value for Case 960
   - Compare with ASHRAE 140 reference implementation
   - May need adjustment based on door opening specifications

3. **Verify Solar Gains Distribution**
   - Ensure solar gains are correctly distributed to zones
   - Check if sunspace is receiving appropriate solar gains
   - Verify solar gains affect thermal network correctly

4. **Review Sensitivity Calculation**
   - Check if sensitivity is correct for multi-zone buildings
   - Verify thermal network algebra for multi-zone case
   - May need to account for inter-zone coupling in sensitivity

## Testing Strategy

1. Run full ASHRAE 140 validation suite to ensure no regressions
2. Add detailed logging to track:
   - HVAC energy per zone (should be 0 for Zone 1)
   - Inter-zone heat transfer rates
   - Free-floating vs controlled temperatures
   - Sensitivity values per zone
3. Compare with reference software (EnergyPlus, ESP-r) if available

## Conclusion

Significant progress has been made on Issue #273:

1. HVAC control is now correctly disabled for free-floating zones
2. Zone-specific thermal parameters are now correctly calculated
3. Window areas are correctly defined for both zones

However, simulation results are still ~20-30x higher than reference values. The remaining issues appear to be related to:
- Temperature update logic (flat back-zone temperature)
- Inter-zone heat transfer calculation
- HVAC sensitivity calculation
- Potential calibration needs for inter-zone conductance

These require deeper investigation into the thermal network algebra and multi-zone physics implementation.

---

**Commits**:
- `38f8f08`: Partial fix for multi-zone HVAC control
- `a68002b`: Implement zone-specific thermal parameters for multi-zone

**Documentation**:
- `docs/ISSUE_273_ROOT_CAUSE_ANALYSIS.md`: Detailed root cause analysis
- `CASE_960_ANALYSIS.md`: Original case specification and analysis
