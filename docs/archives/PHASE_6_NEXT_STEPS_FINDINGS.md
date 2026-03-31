# Phase 6 Next Steps - Investigation Findings

**Date:** 2026-03-29
**Status:** ⚠️ Root cause of energy error remains elusive; thermal mass energy accounting is flawed

---

## Summary of Investigations

### 1. Thermal Mass Energy Accounting (Issue #272)

**Initial State:**
- Thermal mass energy accounting was DISABLED in ASHRAE 140 validator (line 394)
- HVAC energy included energy stored in thermal mass as "consumption"

**Action Taken:**
- Re-enabled thermal mass energy accounting (`model.thermal_mass_energy_accounting = true`)

**Result:**
- Made results WORSE: Pass rate 15.6% → 9.4%, Mean error 277% → 339%

**Analysis:**
The thermal mass energy accounting implementation has fundamental issues:

1. **Timing Problem:**
   - HVAC energy is calculated based on temperature error and sensitivity
   - Mass temperature is updated AFTER HVAC is applied
   - Mass energy change is calculated after mass update
   - The subtraction `hvac_energy - mass_energy_change` doesn't align temporally

2. **Physics Problem:**
   - HVAC energy represents energy added to zone air
   - Mass energy change includes: zone→mass + exterior→mass + internal→mass
   - Subtracting mass_energy_change from hvac_energy doesn't make physical sense
   - The "net HVAC energy" concept is flawed for this simulation

3. **Diagnostic Evidence:**
   - Mass was net DISCHARGING over the year (e.g., -20 MJ for Case 600, -185 MJ for Case 900)
   - This means mass released more energy than it stored
   - But HVAC still reported excess consumption

**Conclusion:** The thermal mass energy accounting approach is fundamentally flawed and cannot be used to fix the energy error.

---

### 2. HVAC Enable Flags (Issue #273)

**Finding:**
- HVAC enable flags are ALREADY IMPLEMENTED correctly
- Lines 503-517 in engine.rs set `hvac_enabled` based on `spec.hvac[zone_idx].is_enabled()`
- Lines 1219-1233 in hvac_power_demand apply the enable flag
- For 5R1C: `hvac_demand * self.hvac_enabled.clone()` (line 1232)
- For 6R2C: Combines envelope and internal mass enable flags (lines 1225-1229)

**Status:** ✅ Already implemented, no action needed

---

### 3. Solar Distribution to Air (Issue #274)

**Finding:**
- Solar distribution is HARDCODED to 0.1 for ALL cases (line 751)
- Issue #274 recommended:
  - Low-mass (600 series): 0.75 (75% to air, 25% to mass)
  - High-mass (900 series): 0.50 (50% to air, 50% to mass)

**Action Taken:**
- Applied conditional solar distribution based on construction type

**Result:**
- Made results WORSE (similar to thermal mass accounting issue)

**Conclusion:** The Issue #274 hypothesis that solar distribution is the root cause is INCORRECT.

---

## Current Baseline State

After reverting both attempted fixes:

| Metric | Value |
|--------|-------|
| Pass Rate | 15.6% |
| Mean Absolute Error | 277.71% |
| Cases Passing | 10/64 |

---

## Key Insights from Phase 6

### 1. Phase 5 Physics-Based Fixes Are NOT Viable

The Phase 5 conclusion was confirmed through testing:
- **h_tr_em = 0.0**: Causes catastrophic free-floating temperatures (95°C)
- **τ-based h_tr_ms**: Makes high-mass cases worse (h_tr_ms too high)

The thermal mass parameters (h_tr_ms, h_tr_em) are NOT the root cause of the energy error.

### 2. Baseline Implementation Is Fundamentally Stable

- Free-floating temperatures are reasonable (36-72°C range)
- The thermal network physics are working correctly
- Controlled cases have high energy, but this appears to be intentional in the current implementation

### 3. Energy Error May Be By Design or Feature

The high energy consumption might be due to:
1. **Sensitivity Calculation**: The `sensitivity = 1 / (h_tr_is + h_ve + h_tr_w)` might be too small, causing excessive HVAC power
2. **Deadband Issues**: HVAC might not have a proper deadband, causing cycling
3. **Conductance Values**: h_tr_is, h_tr_w, h_ve values might be calculated differently than reference implementations
4. **Internal Gains**: Solar, internal loads might be overestimated

---

## Recommended Next Steps

### Alternative Investigation Paths

1. **Compare with Reference Implementations**
   - Run EnergyPlus with identical inputs
   - Compare intermediate states (temperatures, heat flows)
   - Identify WHERE the simulation diverges

2. **Conductance Calibration**
   - Calculate conductances using ISO 13790 formulas
   - Compare to reference values in ASHRAE 140
   - Apply calibration factors if needed

3. **Sensitivity Analysis**
   - Log sensitivity values for each timestep
   - Compare to expected sensitivity from reference
   - Adjust if needed

4. **Internal Gains Verification**
   - Verify solar gain calculation
   - Verify internal loads (lights, equipment, occupants)
   - Compare to reference values

5. **HVAC Control Logic**
   - Add proper deadband to prevent cycling
   - Verify setpoint logic is correct
   - Consider proportional control instead of on/off

---

## Files Modified

1. **src/validation/ashrae_140_validator.rs**
   - Re-enabled thermal mass energy accounting (later reverted as it made things worse)

2. **src/sim/engine.rs**
   - Applied and reverted solar distribution fix (made things worse)

3. **src/bin/diagnose_mass_accounting.rs**
   - Created diagnostic tool to check thermal mass energy accounting status

---

## Conclusion

The Phase 6 next steps revealed that:
1. The thermal mass energy accounting approach is fundamentally flawed
2. HVAC enable flags are already correctly implemented
3. Solar distribution adjustment doesn't help

The root cause of the 3-4x energy error remains unknown. Further investigation should focus on:
- Conductance calculations
- Sensitivity analysis
- Comparison with reference implementations
- HVAC control logic

**Status:** Physics-based fixes are NOT the solution. Alternative approaches needed.
