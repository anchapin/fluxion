# Phase 7A Deeper Analysis Summary

*Date: 2026-03-30*

## Key Findings

### 1. SOLAR-01 Status: PARTIALLY RESOLVED

**Case 600 Peak Cooling Results:**
- Current: 5.77 kW
- Reference: 4.80-6.20 kW
- Status: **WITHIN ACCEPTABLE RANGE** ✅

**Case 900 Peak Cooling Results:**
- Current: 4.57 kW
- Reference: 1.60-2.10 kW
- Status: **OVERPREDICTED** (2x reference)

### 2. CRITICAL NEW ISSUE: Massive Heating Overprediction

**Case 600 Heating Results:**
- Peak Heating: 7.56 kW (Ref: 2.80-3.80 kW) - **2.5x overpredicted**
- Annual Heating: 19.29 MWh (Ref: 5.50-7.50 MWh) - **3.5x overpredicted**

**Case 900 Heating Results:**
- Peak Heating: 7.41 kW (Ref: 1.80-2.40 kW) - **~3x overpredicted**
- Annual Heating: 15.49 MWh (Ref: 1.17-2.04 MWh) - **7-13x overpredicted**

### 3. Annual Cooling Issues

**Case 600 Annual Cooling:**
- Current: 6.50 MWh (Ref: 8.00-10.50 MWh) - **Underpredicted**

**Case 900 Annual Cooling:**
- Current: 4.62 MWh (Ref: 2.13-3.67 MWh) - **Overpredicted**

## Analysis

### Why Peak Cooling is Correct but Heating is Wrong

The fact that Case 600 peak cooling is within reference range while heating is massively overpredicted is a critical finding. This suggests:

1. **Solar distribution fix worked for peak cooling**: Adding solar to phi_ia (air node) with solar_distribution_to_air = 0.7 brought peak cooling within reference range.

2. **Heating issue is separate**: The heating overprediction is NOT caused by solar distribution. Instead, it appears to be a fundamental issue with:
   - Heat loss coefficients (h_tr_em, h_tr_w, h_ve) - may be too high
   - Sensitivity calculation - may be wrong for heating
   - Setpoint application - may be incorrect
   - Heat balance equation - may have sign error or missing term

### Potential Root Causes for Heating Overprediction

1. **Heat loss coefficients too high**: If h_tr_em, h_tr_w, or h_ve are too large, the building would lose heat too rapidly, requiring more heating.

2. **Sensitivity calculation wrong**: Sensitivity = h_ext / (h_ext * (something)...) - if this is calculated incorrectly, heating demand = (setpoint - Ti_free) / sensitivity would be wrong.

3. **HVAC mode determination issue**: If HVAC is in heating mode when it should be off or in cooling mode, it would over-predict heating.

4. **Internal loads too low**: If internal heat gains (lighting, equipment, occupancy) are too low or missing, the building would be colder than expected, requiring more heating.

### Solar Distribution Parameters

**Current values (from Phase 7A investigation):**
- Low-mass (600 series): solar_distribution_to_air = 0.7, solar_beam_to_mass_fraction = 0.3
- High-mass (900 series): solar_distribution_to_air = 0.3, solar_beam_to_mass_fraction = 0.7

This mass-specific approach was tested and showed:
- Case 600 peak cooling improved to within reference
- Case 900 peak cooling still overpredicted

**Key Insight**: The mass-specific solar distribution fix solved peak cooling for low-mass but created new issues for high-mass and exposed a major heating problem.

## Recommendations

### Immediate Actions

1. **Investigate heat loss coefficients**: Verify h_tr_em, h_tr_w, h_ve values are correctly calculated and applied.

2. **Debug sensitivity calculation**: Add diagnostic output to show sensitivity values throughout the simulation.

3. **Check HVAC mode logic**: Verify that heating is not running when it should be cooling or off.

4. **Review internal loads**: Confirm lighting, equipment, and occupancy loads are correctly applied.

### Solar Distribution Strategy

**Recommendation**: Accept that mass-specific solar distribution is not the complete solution for SOLAR-01. Consider:

1. **Dynamic solar distribution**: Vary solar_distribution_to_air based on outdoor temperature or season.

2. **Mass-dependent distribution**: Use different values for different mass classes (already attempted, but may need fine-tuning).

3. **Physics-based approach**: Calculate solar distribution based on first principles rather than empirical tuning.

## Updated Issue Status

| Issue | Previous Status | Current Status | Notes |
|-------|-----------------|-----------------|-------|
| SOLAR-01 (Peak Cooling) | Open | **PARTIALLY RESOLVED** | Low-mass peak cooling within range. High-mass still overpredicted. |
| **NEW**: Heating Overprediction | N/A | **NEW CRITICAL ISSUE** | Annual heating 3-13x overpredicted across all cases. Needs immediate investigation. |

## Next Steps

1. **Priority 1**: Investigate heating overprediction root cause
2. **Priority 2**: Refine solar distribution for high-mass cases
3. **Priority 3**: Update KNOWN_ISSUES.md with new heating issue
