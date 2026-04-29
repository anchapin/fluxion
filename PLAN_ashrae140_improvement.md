# ASHRAE 140 Validation Improvement Plan

## Goal
Increase ASHRAE 140 pass rate from 14.1% to 65%+ using physics-based improvements without calibration factors.

## Current State Analysis

### Validation Results (14.1% Pass Rate)
```
Case 600: Heating=1.60 (Ref: 5.50-7.50) - 77% UNDER   ❌
Case 610: Heating=1.67 (Ref: 4.36-5.79) - 71% UNDER   ❌
Case 620: Heating=1.61 (Ref: 4.50-6.50) - 64% UNDER   ❌
Case 630: Heating=1.65 (Ref: 5.05-6.47) - 67% UNDER   ❌
Case 640: Heating=1.44 (Ref: 2.75-3.80) - 48% UNDER   ❌
Case 650: Cooling=0.62 (Ref: 4.82-7.06) - 91% UNDER  ❌
Case 900: Heating=5.44 (Ref: 1.17-2.04) - 167% OVER  ❌
Case 910: Heating=5.96 (Ref: 1.51-2.28) - 161% OVER  ❌
Case 920: Heating=5.45 (Ref: 3.26-4.30) - 67% OVER   ❌
Case 930: Heating=5.67 (Ref: 4.14-5.34) - 37% OVER   ❌
Case 940: Heating=3.78 (Ref: 0.79-1.41) - 168% OVER  ❌
Case 950: Cooling=5.04 (Ref: 0.39-0.92) - 448% OVER   ❌
Case 960: Heating=9.43 (Ref: 1.65-2.45) - 285% OVER  ❌
```

### Key Observations
1. **600-series (low-mass): SEVERELY UNDER-PREDICTS** - Heating ~1.6 MWh vs ref 5-7 MWh
2. **900-series (high-mass): SEVERELY OVER-PREDICTS** - Heating ~5 MWh vs ref 1-2 MWh
3. **Free-Floating temps: OFF in wrong direction** - Max temps overshoot

---

## Phase 1: Understand Root Causes (Research)

### Task 1.1: Analyze Thermal Mass Implementation
- [ ] Review ISO 13790 5R1C model specification
- [ ] Compare Fluxion's implementation vs standard
- [ ] Identify conductance calculation differences

### Task 1.2: Research EnergyPlus/TRANSYS Methods
- [ ] Study how EnergyPlus handles thermal mass coupling
- [ ] Understand solar gain distribution mechanisms
- [ ] Review infiltration modeling approaches

### Task 1.3: Benchmark Against Known Programs
- [ ] Document what EnergyPlus/DOE-2 produce for each case
- [ ] Identify which programs achieve high pass rates
- [ ] Analyze their thermal model parameters

---

## Phase 2: Thermal Conductance Fixes

### Task 2.1: Fix h_tr_ms Calculation
**Problem**: Current h_tr_ms (mass-to-source conductance) may be incorrect

**ISO 13790 Formula**:
```
h_tr_ms = C_m * h_is / (C_m + C_is)
Where: C_m = effective thermal capacitance (J/K)
       h_is = interior surface conductance (W/K)
```

**Action**: Verify h_tr_ms calculation matches ISO standard:
- [ ] Check effective thermal capacitance formula
- [ ] Verify interior surface conductance (h_is = 4.5 + 3.4 * v, but ≥ 6.1 W/m²K for still air: 8.29 W/m²K per ASHRAE 140)
- [ ] Ensure mass node properly represents building thermal mass

### Task 2.2: Fix h_tr_em Calculation
**Problem**: External-to-mass conductance may not represent heat flow correctly

**ISO 13790 Formula**:
```
h_tr_em = 1 / (1/h_tr_is + 1/h_tr_ms)
Where: h_tr_is = exterior surface conductance + exterior film + windows
```

**Action**:
- [ ] Verify exterior film coefficient (typically 8.29 W/m²K for exterior)
- [ ] Check window heat transfer integration
- [ ] Validate ground coupling handling

### Task 2.3: REMOVE CASE-SPECIFIC h_tr_ms SCALING (CRITICAL!)
**FOUND ISSUE**: Lines 1644-1664 in engine.rs have case-specific tau_scaling:
```rust
let tau_scaling = if spec.case_id.starts_with("9") && ...
{
    if spec.case_id.contains("FF") { 2.0 }
    else if spec.case_id == "900" || ... { 4.0 }  // CASE-SPECIFIC!
    else if spec.case_id == "940" { 4.5 }
    else { 5.0 }
};
h_ms_total /= tau_scaling;
```
**THIS IS EMPIRICAL CALIBRATION TO BE REMOVED!**

### Task 2.4: REMOVE CASE-SPECIFIC h_tr_em SCALING (CRITICAL!)
**FOUND ISSUE**: Lines 1795-1808 in engine.rs have case-specific tau_scaling:
```rust
let tau_scaling = if spec.case_id.starts_with("9") && ...
{
    if spec.case_id.contains("FF") { 1.2 }
    else if spec.case_id == "900" || ... { 1.5 }
    else if spec.case_id == "940" { 1.6 }
    else { 1.8 }
};
h_tr_em_total /= tau_scaling;
```
**THIS IS EMPIRICAL CALIBRATION TO BE REMOVED!**

### Summary of Empirical Calibrations to Remove
1. **h_tr_ms tau_scaling** (lines 1644-1664): 2.0x-5.0x case-specific
2. **h_tr_em tau_scaling** (lines 1795-1808): 1.2x-1.8x case-specific
3. **peak_calibration** (REMOVED ✓)
4. **time_constant_sensitivity_correction** (=1.0 already ✓)

### Task 2.3: Fix Thermal Capacitance (Cm)
**Problem**: Effective thermal capacitance may be wrong

**ISO 13790 Reference Values**:
| Construction | C_m (J/m²K) |
|--------------|------------|
| Light (600)  | 55,000     |
| Medium       | 110,000    |
| Heavy (900)   | 370,000    |

**Action**:
- [ ] Calculate Cm from construction layers (not assumed values)
- [ ] Verify area-weighted calculation for multiple surfaces
- [ ] Check internal mass contribution

---

## Phase 3: Solar Gain Distribution

### Task 3.1: Fix Solar Fraction to Mass
**Problem**: All solar going directly to air vs stored in mass

**ISO 13790**: Fraction solar to air = 0.1 × (1 - f_ms) + f_ms
- Where f_ms ≈ 0.8 for heavy mass, ~0.4 for light mass

**Action**:
- [ ] Implement proper solar fraction calculation
- [ ] Ensure mass absorbs appropriate fraction
- [ ] Track solar contribution to each node

### Task 3.2: Fix Direct vs Diffuse Split
- [ ] Verify window solar transmission
- [ ] Check shading coefficient application
- [ ] Validate irradiance calculation

---

## Phase 4: Infiltration & Ventilation

### Task 4.1: Fix Infiltration Model
**Problem**: Infiltration calculation may not match ASHRAE 140 specification

**ASHRAE 140 Specification**:
- [ ] Verify ACH (Air Changes per Hour) conversion
- [ ] Check conductivity formula: h_ve = ACH × V × ρ × cp / 3600
- [ ] Where: ρ = 1.2 kg/m³, cp = 1005 J/kg·K

### Task 4.2: Fix Ground Coupling
- [ ] Verify floor-on-ground vs exposed floor handling
- [ ] Check ground temperature input
- [ ] Fix h_tr_floor conductance calculation

---

## Phase 5: Temperature Calculation Fixes

### Task 5.1: Fix Time Step Integration
- [ ] Verify Euler integration stability
- [ ] Check temperature update for mass node
- [ ] Validate energy balance at each step

### Task 5.2: Fix Free-Floating Temperature Calculation
**Problem**: FF cases produce wrong temperature swings

**Root Cause Analysis**:
- [ ] Verify thermal mass response timeconstant τ
- [ ] Check solar gain absorption
- [ ] Validate heat flow to exterior, ground, adjacent zones

---

## Phase 6: HVAC Model Verification

### Task 6.1: Fix Sensitivity Calculation
- [ ] Verify heating/cooling sensitivity: q_sen = (T_i - T_set) × H
- [ ] Check how H (sensible coefficient) is calculated
- [ ] Verify ideal loads system vs energy-based calculation

### Task 6.2: Fix HVAC Energy Tracking
- [ ] Verify energy integration from power
- [ ] Check peak detection logic
- [ ] Validate annual energy accumulation

---

## Phase 7: Reference Data Validation

### Task 7.1: Verify Benchmark Data
Reference ranges come from multiple programs. Key questions:
- [ ] Confirm 600-series reference: EnergyPlus range ~5.5-7.5 MWh
- [ ] Confirm 900-series reference: EnergyPlus ~1.2-2.0 MWh
- [ ] Check if ranges represent ALL programs including DOE-2

### Task 7.2: Understand Range Targets
The reference ranges span multiple programs:
- Best performers achieve ~within ±5%
- Typical programs achieve ±10-20%
- ASHRAE 140 allows ±30% band for acceptance

**Target**: Without case-specific tuning, achieve 65% pass rate

---

## Implementation Priority

Based on magnitude of errors:

### Critical (Immediate)
1. **Task 2.1**: Fix h_tr_ms - affects ALL cases
2. **Task 2.3**: Fix Cm calculation - affects thermal mass timing
3. **Task 3.1**: Fix solar to mass fraction - fixes over-prediction in 900-series

### High Priority
4. **Task 4.1**: Fix infiltration conductance - affects 600-series
5. **Task 5.1**: Fix integration stability - baseline accuracy

### Medium Priority
6. **Task 6.1**: Verify HVAC sensitivity - improves all energy values
7. **Task 7.1**: Verify reference values - confirm targets are correct

---

## Testing Strategy

### Test Each Fix Independently
```
test_thermal_conductance()
  - Run Case 600, 900
  - Check heating/cooling values
  - Verify direction (over/under)

test_solar_distribution()
  - Run with solar gains
  - Verify temperature swing

test_free_floating()
  - Run FF cases
  - Compare min/max temps
```

### Measure Progress
- Check pass rate after each phase
- Target: 35% after Phase 2, 50% after Phase 4, 65% after Phase 7

---

## Risk Mitigation

### Temporary Regressions Expected
- Some fixes may temporarily worsen certain cases
- Track which cases worsen per fix
- Accept trade-offs for long-term physics accuracy

### Rollback Strategy
- Keep git commits organized by fix
- Can revert if overall pass rate drops below 10%

---

## Timeline Estimate

| Phase | Description | Estimated Effort |
|-------|-------------|-------------------|
| 1 | Research & Analysis | 1-2 weeks |
| 2 | Thermal Conductance | 2-3 weeks |
| 3 | Solar Distribution | 1-2 weeks |
| 4 | Infiltration/Ventilation | 1 week |
| 5 | Temperature Calculation | 1 week |
| 6 | HVAC Verification | 1 week |
| 7 | Validation & Tuning | 2-3 weeks |
| **Total** | | **9-15 weeks** |

---

## Key Resources

### EnergyPlus Documentation
- Engineering Reference: Section on Thermal Models
- Input Output Reference: HVAC objects

### ASHRAE 140 Standard
- Section 5: Test Cases for Building Thermal Fabric
- Section 6: Test Cases for HVAC Equipment

### ISO 13790 Standard
- Annex A: Simplified methods for calculation
- 5R1C model specification

### Previous Validation Reports
- SBEED Validation Studies
- IBPSA Building Thermal Fabric Updates

---

## Tools for Research

### Wolfram Alpha API
Use for:
- Thermal property calculations
- Unit conversions
- Heat transfer equation solving

### Context7
Search for:
- EnergyPlus thermal model documentation
- ISO 13790 implementation details

### Web Search
Target:
- ASHRAE 140 methodology papers
- EnergyPlus vs TRNSYS comparison
- Thermal model validation studies
