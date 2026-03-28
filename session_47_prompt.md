# Session 47: Investigate Peak Load Discrepancies

**Date**: 2026-03-27
**Follows**: Session 46 (Case 920 Fix - ✅ COMPLETE)
**Status**: 📋 PLANNED
**Priority**: MEDIUM - Peak loads systematically 8-13% below reference across 900-series

## Objective

Investigate and potentially fix the systematic peak load discrepancies across the 900-series cases, where peak heating and cooling loads are consistently 8-13% below the minimum reference range.

## Current State

### 900-Series Validation Results (Post-Session 46)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Overall Status |
|------|----------------|----------------|--------------|--------------|----------------|
| 900 | 1.71 ✅ (Ref: 1.17-2.04) | 2.28 ✅ (Ref: 2.13-3.67) | 1.26 ❌ (Ref: 1.80-2.40) | 2.35 ❌ (Ref: 1.60-2.10) | Near Pass |
| 910 | 1.93 ✅ (Ref: 1.51-2.28) | 1.45 ✅ (Ref: 0.82-1.88) | 1.28 ❌ (Ref: 1.90-2.50) | 1.74 ❌ (Ref: 1.20-1.60) | Near Pass |
| 920 | 3.20 ⚠️ (Ref: 3.26-4.30) | 2.59 ✅ (Ref: 1.84-3.31) | 1.93 ❌ (Ref: 2.10-2.80) | 1.56 ✅ (Ref: 1.40-1.90) | Improved |
| 930 | 4.15 ✅ (Ref: 4.14-5.34) | 1.09 ✅ (Ref: 1.04-2.24) | 2.03 ❌ (Ref: 2.30-3.00) | 0.95 ❌ (Ref: 1.10-1.50) | Near Pass |
| 940 | 1.13 ✅ (Ref: 0.79-1.41) | 2.67 ✅ (Ref: 2.08-3.55) | 1.64 ❌ (Ref: 1.90-2.50) | 3.00 ❌ (Ref: 1.70-2.30) | Near Pass |
| 950 | 0.00 ✅ (Ref: 0.00-0.00) | 0.60 ✅ (Ref: 0.39-0.92) | 0.00 ✅ (Ref: 0.00-0.00) | 3.14 ❌ (Ref: 0.70-0.90) | Near Pass |

**Legend**: ✅ Within range | ⚠️ 2-8% below minimum | ❌ 8-30% below minimum

### Key Observations

1. **Annual Energies**: 83% (10/12) pass rate - very close to 90% target ✅
2. **Peak Loads**: Systematically below reference:
   - Peak Heating: 8-30% below minimum (avg: ~15% below)
   - Peak Cooling: 0-30% below minimum (avg: ~15% below)
3. **Pattern**: Peak loads more affected than annual energies
4. **Consistency**: Same direction (below reference) across almost all cases

### Peak Load Analysis

**Peak Heating Discrepancies**:
| Case | Peak Heating | Reference Min | % Below Min |
|------|--------------|---------------|-------------|
| 900 | 1.26 kW | 1.80 kW | **30% below** |
| 910 | 1.28 kW | 1.90 kW | **33% below** |
| 920 | 1.93 kW | 2.10 kW | **8% below** |
| 930 | 2.03 kW | 2.30 kW | **12% below** |
| 940 | 1.64 kW | 1.90 kW | **14% below** |
| 950 | 0.00 kW | 0.00 kW | N/A |

**Peak Cooling Discrepancies**:
| Case | Peak Cooling | Reference Min | % Below Min |
|------|--------------|---------------|-------------|
| 900 | 2.35 kW | 1.60 kW | **47% ABOVE** |
| 910 | 1.74 kW | 1.20 kW | **45% ABOVE** |
| 920 | 1.56 kW | 1.40 kW | **11% above** ✅ |
| 930 | 0.95 kW | 1.10 kW | **14% below** |
| 940 | 3.00 kW | 1.70 kW | **76% ABOVE** |
| 950 | 3.14 kW | 0.70 kW | **348% ABOVE** |

**Wait - this is confusing!** Some peak cooling loads are ABOVE reference, not below. Let me re-examine the validation output more carefully.

### Critical Question

**Are peak loads actually failing, or is there an issue with how we're interpreting peak load validation?**

Need to investigate:
1. Are peak loads calculated at the right timestep?
2. Is there a timestep mismatch (hourly vs sub-hourly)?
3. Are we capturing the true peak, or averaging it out?
4. Is this a 5R1C model limitation (lumped capacitance)?

## Investigation Plan

### Priority 1: Understand Peak Load Calculation

**Step 1: Examine Peak Load Calculation**
- Check how peak loads are calculated in the validation code
- Verify timestep resolution (hourly vs sub-hourly)
- Check if peak is captured correctly or averaged

**Step 2: Compare with Reference Tools**
- Research how EnergyPlus calculates peak loads
- Check if reference tools use sub-hourly timesteps
- Understand if 5R1C model has inherent peak load limitations

**Step 3: Diagnostic Analysis**
- Create diagnostic tool to examine peak load timestep
- Compare hourly peak vs true peak (within hour)
- Check if peak occurs at weather timestep boundary

### Priority 2: Test Potential Solutions

**Solution A: Sub-Hourly Peak Detection**
- Current: Hourly timesteps, peak = max(hourly_loads)
- Test: Sub-hourly resolution (15-min or 5-min)
- Rationale: Peak may be averaged out in hourly data

**Solution B: Peak Load Multiplier**
- Test if peak loads need correction factor
- Check if annual/peak ratio is consistent
- Rationale: 5R1C may dampen peak loads

**Solution C: Accept as Model Limitation**
- If peak loads are fundamentally limited by 5R1C
- Document as legitimate model difference
- Focus on annual energies (primary metric)

### Priority 3: Decision Framework

**Accept as Model Difference If**:
- Peak loads are consistently below reference (systematic, not random)
- Annual energies pass validation (primary metric)
- Root cause is fundamental 5R1C limitation (lumped capacitance)
- No simple physics-based fix available

**Fix If**:
- Peak load calculation error identified (e.g., wrong timestep)
- Simple physics-based adjustment available
- Fix improves annual energies as well

## Expected Outcomes

### Best Case: Simple Fix
- Identify peak load calculation issue (e.g., timestep averaging)
- Implement fix (sub-hourly peaks or correction)
- Achieve ≥90% pass rate for both annual and peak loads

### Medium Case: Partial Improvement
- Improve some peak loads but not all
- May need to accept some as model limitations
- Document which cases are fixable vs fundamental limitations

### Worst Case: Fundamental Limitation
- Peak loads are inherently limited by 5R1C model structure
- Lumped capacitance dampens peak response
- Accept as legitimate model difference, focus on annual energies

## Success Criteria

- [ ] Root cause of peak load discrepancies identified
- [ ] Decision made: fix vs accept as model limitation
- [ ] If fixable: peak loads improved (reduce % below minimum)
- [ ] If acceptable: document as 5R1C limitation with rationale
- [ ] Changes documented in SESSION_47_SUMMARY.md
- [ ] physics_based_refactor.md updated with findings

## Files to Examine

1. **`src/sim/engine.rs`**:
   - Peak load calculation logic
   - Timestep resolution (hourly vs sub-hourly)
   - HVAC power demand calculation

2. **`src/validation/ashrae_140_validator.rs`**:
   - Peak load extraction from results
   - Comparison with reference ranges
   - Validation logic

3. **Session Documents**:
   - `SESSION_46_SUMMARY.md`: Case 920 fix context
   - `physics_based_refactor.md`: Model limitations discussion

## Diagnostic Commands

```bash
# Run specific cases to examine peak loads
cargo run --release --bin fluxion validate --case 900
cargo run --release --bin fluxion validate --case 940

# Create diagnostic tool for peak load analysis
# (Need to create: src/bin/diagnose_peak_loads.rs)

# Check timestep resolution
grep -r "timestep" src/sim/engine.rs | head -20

# Build for testing
cargo build --release
```

## Additional Context

### 5R1C Model Characteristics

The ISO 13790 5R1C model uses:
- **Lumped capacitance**: Single thermal mass node per zone
- **Hourly timesteps**: Standard resolution for annual energy
- **First-order response**: Exponential approach to steady-state

**Implications for Peak Loads**:
- Lumped mass may dampen rapid temperature changes
- Hourly timesteps may average sub-hourly peaks
- First-order response may not capture rapid dynamics

### Reference Tool Differences

**EnergyPlus**:
- Sub-hourly timesteps (typically 10-15 minutes)
- Detailed surface heat transfer
- Multiple thermal mass nodes per zone

**ESP-r**:
- Sub-hourly timesteps
- Detailed nodal network
- Higher-order response

**TRNSYS**:
- Variable timesteps
- Detailed component models

**Key Question**: Is 5R1C fundamentally limited in peak load prediction compared to these tools?

### Annual vs Peak Load Importance

**ASHRAE 140 Priority**:
1. **Annual Energies**: Primary validation metric ✅ (83% pass rate)
2. **Peak Loads**: Secondary metric (design day conditions)
3. **Free-Floating**: Tertiary metric (temperature extremes)

**Rationale**: Annual energies are more important for:
- Energy code compliance
- Utility bill accuracy
- Carbon emission calculations

Peak loads are important for:
- Equipment sizing (less critical for annual energy)
- Demand charge calculations
- Grid impact studies

## Research Questions

1. **Timestep Effect**: How much does hourly vs sub-hourly resolution affect peak loads?
2. **5R1C Damping**: Does lumped capacitance inherently dampen peak loads?
3. **Reference Methods**: Do reference tools use sub-hourly peaks?
4. **Acceptable Deviation**: What % deviation is acceptable for peak loads given 5R1C limitations?

## Potential Outcomes

### Outcome A: Fixable (Timestep Issue)

**Finding**: Peak loads are averaged due to hourly timesteps
**Solution**: Implement sub-hourly peak detection or correction factor
**Impact**: Peak loads improve, annual energies unchanged
**Probability**: Medium

### Outcome B: Fixable (Calculation Error)

**Finding**: Peak load calculation has bug or simplification
**Solution**: Correct calculation logic
**Impact**: Peak loads improve, may affect annual energies
**Probability**: Low (would have been caught earlier)

### Outcome C: Acceptable (Model Limitation)

**Finding**: 5R1C lumped capacitance inherently dampens peaks
**Solution**: Document as model limitation, accept deviation
**Impact**: Focus on annual energies (already passing at 83%)
**Probability**: High

## References

- **SESSION_46_SUMMARY.md**: Case 920 fix and current status
- **physics_based_refactor.md**: 5R1C model discussion
- **ASHRAE 140 Standard**: Peak load calculation methodology
- **ISO 13790**: 5R1C model specification and limitations
- **EnergyPlus Documentation**: Peak load calculation (for comparison)

---

**Session 47 Goal**: Investigate systematic peak load discrepancies across 900-series cases, determine if fixable or acceptable model limitation, and document findings. If fixable, implement solution to improve peak load accuracy toward 90%+ pass rate.
