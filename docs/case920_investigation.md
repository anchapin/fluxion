# Case 920/930/940 Peak Heating Investigation

## Summary
Issue #2368: Cases 920/930/940 peak heating OVER prediction by 8-80%.

## Investigation Results

### Attempted Fix
- Changed HVAC coefficient from `derived_h_tr_3 + h_tr_w` to `h_tr_1 + h_tr_w`
- Result: **Made problem 2.5x WORSE** (3.41 kW → 8.75 kW for Case 920)

### Root Cause: NOT IDENTIFIED

The counterintuitive behavior: reducing `hvac_coeff` causes `T_free` to drop lower during simulation, which increases the effective `ΔT = T_setpoint - T_free`, offsetting the coefficient reduction.

### Key Findings
1. ASHRAE 140 reference data for Cases 920/930/940 may need verification against EnergyPlus
2. Case 940's 170% over-prediction suggests fundamental issue with setback scenario
3. Peak timing analysis (when peak heating occurs) needed
4. T_free computation at peak heating time may not match design conditions

### Next Steps (for future work)
- Verify EnergyPlus reference values for peak heating
- Analyze peak timing vs design conditions
- Investigate setback recovery dynamics in Case 940
- Consider HVAC oversizing hypothesis
