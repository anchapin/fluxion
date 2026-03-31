# Phase 8B: Thermal Network Root Cause Identified

## Executive Summary

**Status**: 🔍 Root Cause Identified

Fluxion's systematic 60-90% energy under-prediction is caused by **empirical sensitivity calibration** that creates an artificially high thermal resistance (144.78 K/W), preventing realistic HVAC power demand.

## Evidence

### EnergyPlus vs Fluxion Comparison

| Case | EP Heating (MWh) | Fluxion Heating (MWh) | Error | EP Cooling (MWh) | Fluxion Cooling (MWh) | Error |
|-------|------------------|----------------------|--------|------------------|----------------------|--------|
| 600 | 15,569 | 6.10 | 61% under | 21,759 | 6.55 | 70% under |
| 610 | 15,750 | 6.19 | 61% under | 15,648 | 4.91 | 69% under |
| 620 | 16,138 | 5.25 | 67% under | 14,667 | 2.80 | 81% under |
| 630 | 17,208 | 5.53 | 68% under | 10,249 | 1.56 | 85% under |
| 900 | 5,980 | 1.74 | 71% under | 8,993 | 3.77 | 58% under |
| 910 | 7,030 | 1.91 | 73% under | 5,002 | 2.75 | 45% under |

### Debug Output (Case 600, Hour 12)
```
h_ext = 105.54 W/K
h_is_m = 112.11 W/K
h_total = 144.78 W/K
sensitivity = 0.006907 K/W (thermal resistance = 144.78 K/W)
```

**Physical Meaning**: 144.78 K/W thermal resistance means 1W of HVAC power changes temperature by only 0.007K - **physically impossible for normal operation**.

## Root Cause Analysis

### Issue 1: Empirical Sensitivity Calibration (Primary)

**Location**: `src/sim/engine.rs:2439-2463`

**Current Implementation**:
```rust
// Empirical weights to "calibrate against ASHRAE 140 reference values"
let h_is_m = self.derived_h_ms_is_prod.clone() / self.derived_term_rest_1.clone();
let h_with_mass = self.derived_h_ext.clone() + h_is_m.clone();
let is_low_mass = self.case_id.starts_with('6') || self.case_id == "195";
let h_total = if is_low_mass {
    self.derived_h_ext.clone() * 0.65 + h_with_mass.clone() * 0.35
} else {
    self.derived_h_ext.clone() * 0.75 + h_with_mass.clone() * 0.25
};
self.derived_sensitivity = self.temperatures.constant_like(1.0) / h_total.clone();
```

**Problem**:
- Weights (0.65/0.35, 0.75/0.25) are ad-hoc calibrations, not derived from physics
- Comments explicitly state: "Based on calibration against ASHRAE 140 reference values"
- This masks underlying physics errors rather than fixing them

**Impact**:
- Creates artificially high thermal resistance (144.78 K/W)
- Suppresses HVAC power demand → Systematic under-prediction (60-90%)
- Makes building appear thermally "tight" from interior perspective

### Issue 2: Incorrect h_tr_is Calculation

**Location**: `src/sim/engine.rs:2361`

**Current Implementation**:
```rust
const H_SI: f64 = 3.45; // W/m²K - ASHRAE 140 simplified 5R1C value
// h_tr_is = 3.45 × (opaque_area + floor_area × 2.0)
```

**Problem**:
- 3.45 W/m²K is significantly lower than ASHRAE 140 spec:
  - Default: 8.29 W/m²K
  - Walls: 7.69 W/m²K
  - Ceilings: 10.0 W/m²K
  - Floors: 5.88 W/m²K
- Uses `zone_area * 2.0` as proxy for ceiling + floor (not accurate for h_tr_is)

**Impact**:
- Underestimates interior heat transfer
- Contributes to overall thermal resistance being too high
- Makes building require less HVAC energy to maintain temperature

## Why This Causes 60-90% Under-Prediction

The thermal network equation shows the problem:

```
sensitivity = 1 / h_total
         where h_total = h_ext + h_is_m
```

When sensitivity is too small (h_total too large):
- HVAC power needed to change temperature = ΔT / sensitivity = ΔT × (h_ext + h_is_m)
- Small sensitivity → Large h_total → Small HVAC power → Less energy consumption
- Less energy accumulation → Under-prediction

## Required Fixes

### Fix 1: Use ASHRAE 140 Surface-Specific h_tr_is

**Action**: Update `h_tr_is` calculation to use surface-specific ASHRAE 140 coefficients:
```rust
// Current (incorrect):
self.h_tr_is = 3.45 × (opaque_wall_area + zone_area * 2.0);

// Proposed (correct):
let wall_h_tr_is = INTERIOR_FILM_COEFF_WALL * wall_area;  // 7.69 W/m²K
let ceiling_h_tr_is = INTERIOR_FILM_COEFF_CEILING * ceiling_area;  // 10.0 W/m²K
let floor_h_tr_is = INTERIOR_FILM_COEFF_FLOOR * floor_area;  // 5.88 W/m²K
self.h_tr_is = wall_h_tr_is + ceiling_h_tr_is + floor_h_tr_is;
```

**Expected Impact**:
- h_tr_is increases from ~3.45 × 171.6 = 609 W/K to ~8 × 171.6 = 1373 W/K (2.3×)
- This increases h_is_m significantly
- Reduces thermal resistance → Increases HVAC power demand
- Moves results toward EnergyPlus values

**Risk**: If other conductances are also incorrect, this may over-predict. The empirical weights were compensating.

### Fix 2: Remove Empirical Weights

**Action**: Replace empirical weighting with physics-based calculation:
```rust
// Current (empirical):
let h_total = h_ext * 0.65 + (h_ext + h_is_m) * 0.35;

// Proposed (physics-based):
let h_total = h_ext + h_is_m;  // No weighting
self.derived_sensitivity = self.temperatures.constant_like(1.0) / h_total;
```

**Expected Impact**:
- Sensitivity based on actual thermal resistances
- May initially show worse results if other conductances are also incorrect
- Provides baseline for identifying which conductances need adjustment
- Necessary for long-term physics correctness

### Fix 3: Audit All Conductances

**Required Actions**:
1. Validate h_tr_em (exterior → mass) uses construction U-values
2. Validate h_tr_w (windows) uses window U-value and area
3. Validate h_ve (ventilation) uses infiltration rate and volume
4. Validate h_tr_is (interior surfaces) uses ASHRAE 140 surface-specific values
5. Validate h_tr_ms (mass → surface) uses construction layer properties
6. Validate thermal capacitance calculation

### Fix 4: Extract Hourly Power Traces

**Action**: Modify `extract_ep_reference.py` to extract hourly HVAC power from EnergyPlus and compare with Fluxion

**Purpose**: Determine if under-prediction is:
- Uniform across all conditions (sensitivity issue)
- Concentrated in specific conditions (heat transfer path issue)
- Time-of-day mismatch (thermal mass dynamics issue)

## Implementation Priority

1. **HIGH**: Implement Fix 1 (AShRAE 140 h_tr_is) - Will show immediate improvement
2. **HIGH**: Implement Fix 2 (Remove empirical weights) - Required for physics correctness
3. **MEDIUM**: Conduct audit (Fix 3) - Necessary baseline
4. **MEDIUM**: Extract hourly data (Fix 4) - For diagnosis

## Files to Modify

1. `src/sim/engine.rs`:
   - Update `update_derived_parameters()` to use ASHRAE 140 surface-specific h_tr_is
   - Remove empirical weighting (lines 2449-2463)
   - Add geometry fields for separate surface areas if needed

2. `src/sim/construction.rs`:
   - Consider removing hardcoded `H_SI = 3.45` constant
   - Use ASHRAE 140 constants from `INTERIOR_FILM_COEFF_*`

3. `tools/extract_ep_reference.py`:
   - Add hourly HVAC power extraction
   - Extract power at each timestep for comparison

## Testing Plan

### Test 1: Before/After Fix 1
```bash
cargo build --release
./target/release/fluxion validate -c 600
# Compare: Fluxion heating should increase from 6.10 to 8-14 MWh (34-133% increase)
```

### Test 2: After Fix 2 (if still under-predicted)
- Compare physics-based sensitivity vs empirical calibration
- Identify which conductances need adjustment

## Success Criteria

- [ ] Fix 1: h_tr_is uses ASHRAE 140 surface-specific coefficients
- [ ] Fix 2: Empirical weights removed, using physics-based sensitivity
- [ ] Fix 3: All conductances validated against ASHRAE 140
- [ ] Fix 4: Hourly power traces extracted and compared
- [ ] Case 600 heating: 8-15 MWh (within 20% of 15.57 MWh)
- [ ] Case 600 cooling: 7-20 MWh (within 20% of 21.76 MWh)

## Next Actions

1. Review this analysis
2. Prioritize Fix 1 (h_tr_is ASHRAE 140) - highest impact
3. Implement Fix 2 (remove empirical weights)
4. Test and validate
5. Document findings in KNOWN_ISSUES.md

**Status**: Root cause identified, fixes designed, ready for implementation.
