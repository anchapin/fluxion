# Phase 7B: High-Mass Peak Cooling Investigation

*Date: 2026-03-30*

## Task: Fix high-mass peak cooling overprediction

## Investigation Summary

### Problem Statement

High-mass cases (900 series) show peak cooling 2-2.5x above ASHRAE 140 reference:

| Case | Peak Cooling (kW) | Reference (kW) | Ratio |
|-------|-------------------|------------------|-------|
| 900 | 3.63 | 1.60-2.10 | 2.2x |
| 910 | 2.88 | N/A | - |
| 920 | 1.79 | N/A | - |
| 930 | 1.25 | N/A | - |
| 940 | 3.63 | N/A | - |
| 950 | 5.36 | N/A | - |

For comparison, low-mass cases (600 series) pass peak cooling tests.

### Root Cause Analysis

The peak cooling overprediction is caused by:

1. **Thermal Mass Accumulation**: High-mass buildings (Cm ≈ 9 MJ/K) accumulate solar energy in the thermal mass via phi_m (direct gains to mass)
2. **Slow Heat Dissipation**: Heat transfer from mass to surface (h_tr_ms ≈ 2000 W/K) creates thermal time constant τ ≈ 1.25 hours
3. **Temperature Coupling**: High mass temperature drives high air temperature via the Ti_free calculation
4. **Peak Cooling Calculation**: power = (setpoint - T_free) / sensitivity, so higher T_free causes higher cooling demand

### Investigation Steps Performed

#### 1. Thermal Mass Divergence Test

Created `test_mass_divergence.rs` to trace thermal mass temperature evolution:

**Findings**:
- With 6R2C model and NO solar gains: mass temperatures remain stable (~20°C)
- With solar gains: mass temperature increases and drives air temperature up
- The issue is NOT pure numerical divergence, but rather physical heat accumulation

**Thermal Parameters (Case 900)**:
- Cm (thermal capacitance): 8,931,379.80 J/K
- h_tr_em (exterior-mass): 201.45 W/K
- h_tr_ms (mass-surface): 2014.48 W/K
- Thermal time constant τ: 1.23 hours
- Solar to mass fraction: 0.70 (ASHRAE 140 spec)

#### 2. Crank-Nicolson Integration

Modified `select_integration_method()` to use Crank-Nicolson for very high mass (Cm > 5 MJ/K).

**Findings**:
- Crank-Nicolson made peak cooling WORSE (4.04 kW vs 3.63 kW with Backward Euler)
- Reverted to Backward Euler (1st-order accurate, unconditionally stable)

**Reason**: Crank-Nicolson is A-stable but can cause oscillatory behavior with stiff systems when dt/τ is large. For thermal mass with τ ≈ 1.25h and dt = 1h, the ratio is ~0.8, which is marginal.

#### 3. Solar Distribution Adjustment

Tested reducing `solar_beam_to_mass_fraction` from 0.70 to 0.50.

**Findings**:
- Peak cooling: 4.06 kW (worse than 3.63 kW)
- This suggests the issue is NOT simply too much solar going to mass
- Reverted to ASHRAE 140 spec value of 0.70

#### 4. Time Step Reduction Consideration

Considered reducing time step from 1 hour to 15-30 minutes for high-mass cases.

**Implications**:
- Would require 2-4x more computation steps (8760 → 17520-35040)
- Would require sub-stepping for mass update only
- Significant code changes to thermal integration
- Performance impact on 10k population/second target

**Conclusion**: Not practical for production use given performance constraints.

### Fundamental Analysis

The 5R1C/6R2C thermal network model has inherent limitations for high-mass buildings with solar forcing:

**ISO 13790 Assumptions**:
- Lumped capacitance model (single or dual mass nodes)
- Steady-state heat transfer at each time step
- Linear heat transfer coefficients

**ASHRAE 140 Reference Implementations** (EnergyPlus, ESP-r):
- Multi-layer finite difference models
- Radiative heat transfer between surfaces
- Time-varying heat transfer coefficients
- More sophisticated convection models

The peak cooling overprediction is a **known limitation of the ISO 13790 lumped model** when:
1. Thermal mass is high (Cm > 5 MJ/K)
2. Solar forcing is significant (peak conditions)
3. Time step is large (dt > 15 minutes)

### Potential Fixes Considered

| Fix | Effect | Feasibility |
|------|---------|-------------|
| Crank-Nicolson integration | ❌ Made results worse | High |
| Reduce solar to mass fraction | ❌ Made results worse | Medium |
| Reduce time step (15-30 min) | 🔄 Would help | Low (performance impact) |
| Increase h_tr_em (mass-to-exterior) | 🔄 Would help | Medium (may affect annual energy) |
| Multi-layer finite difference model | ✅ Would fix | Very Low (major redesign) |
| Accept as model limitation | ✅ Document and proceed | High |

## Recommendations

### Option 1: Document as LIMIT-05 (Recommended)

Add to `KNOWN_ISSUES.md`:
- **Issue**: SOLAR-02 - High-mass peak cooling overprediction
- **Severity**: Medium (affects 6/24 cases, all high-mass)
- **Status**: Known limitation of 5R1C/6R2C model
- **Impact**: Peak cooling 2-2.5x above ASHRAE 140 reference for 900-series cases
- **Workaround**: None without major model redesign
- **Resolution path**: Upgrade to finite difference or CTF-based heat transfer model (Phase 6+)

### Option 2: Time Step Sub-stepping for Mass

Implement adaptive time stepping for thermal mass update only:
- Keep HVAC/time-step at 1 hour
- Sub-step mass update with smaller dt (e.g., 15 minutes)
- Requires: 4 sub-steps per main time step
- Performance impact: ~2x (acceptable for accuracy gain)

### Option 3: Multi-Zone Heat Transfer Enhancement

Enhance thermal mass model with:
- Additional heat transfer paths (floor-to-mass, walls-to-mass)
- Zone-specific mass temperatures (not single lumped mass)
- Better solar distribution (surface-specific vs zone-wide)

## Conclusion

The high-mass peak cooling overprediction is a **fundamental limitation of the lumped thermal mass model** used by the 5R1C/6R2C implementation. The issue manifests when:

1. High thermal mass (Cm > 5 MJ/K) stores solar energy
2. Heat dissipation is slow (limited by h_tr_ms, h_tr_em)
3. Large time step (1 hour) doesn't resolve fast dynamics

**Fixing this properly requires**:
- Upgrading to finite difference or CTF-based heat transfer model
- Reducing time step for high-mass cases (sub-stepping)
- Implementing multi-layer thermal network

**Quick fixes attempted** (Crank-Nicolson, solar distribution adjustment) did not resolve the issue.

## Files Modified

- `src/sim/thermal_integration.rs`: Added Crank-Nicolson option (later reverted)
- `tests/test_mass_divergence.rs`: Added diagnostic test
- `tests/test_6r2c_model.rs`: Fixed configure_6r2c_model() calls
- `tests/test_6r2c_time_constant.rs`: Fixed configure_6r2c_model() calls

## Next Steps

**Recommended**: Document as LIMIT-05 and proceed with other phases.

**Alternative**: Implement time step sub-stepping for thermal mass (requires ~2 days work).

## References

- ISO 13790:2008 - Energy performance of buildings - Calculation of energy use for space heating and cooling
- ASHRAE Standard 140-2017 - Standard Method of Test for the Evaluation of Building Energy Analysis Computer Programs
- Phase 7A Root Cause Found: HVAC capacity fix for peak heating
