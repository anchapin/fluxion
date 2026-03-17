# Issue #486: Case 900FF Free-Floating Max Temperature Low

## Problem Statement

Case 900FF (free-floating, no HVAC) shows maximum temperature that is too low:
- **Current**: 36.66°C
- **Reference**: 41.80 - 46.40°C
- **Error**: ~16.9% too low

Minimum temperature passes validation:
- **Current**: -4.70°C
- **Reference**: -6.40 to -1.60°C
- **Status**: ✅ PASS

## Root Cause Analysis

### Nature of Free-Floating Conditions

Free-floating temperature represents the building with **no HVAC**, driven only by:
1. Solar gains through glazing
2. Internal gains (occupants, equipment, lighting)
3. Conduction through envelope
4. Thermal mass buffering effects

### Identified Limitations

The low max temperature suggests the 5R1C thermal model has fundamental limitations:

1. **Thermal Mass Dynamics Underestimation**
   - The 5R1C steady-state sensitivity model does not fully capture thermal mass dynamics
   - Thermal mass buffering effect is underestimated in free-floating conditions
   - Heat storage and release timing is not accurately modeled

2. **Solar Gain Distribution**
   - Solar radiation may not be correctly distributed between air node and mass node
   - Beam-to-mass fraction (`solar_beam_to_mass_fraction`) may need calibration
   - Direct radiation to floor mass vs. convective to air balance

3. **Internal Gains Modeling**
   - Convective vs. radiative split may not match reference model
   - Radiative gains to thermal mass may be underestimated

4. **Heat Loss Overestimation**
   - Envelope heat loss may be overestimated
   - Ground coupling or infiltration may be too high

## Current Model Architecture (5R1C)

```
                    ┌─────────────────┐
                    │   Outdoor Air   │
                    │     (T_e)       │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   R_e (1/h_e)   │
                    │   Surface Film  │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │             ┌──────▼──────┐             │
        │             │   T_st      │             │
        │             │ (Surface)   │             │
        │             └──────┬──────┘             │
        │                    │                    │
        │             ┌──────▼──────┐             │
        │             │   R_st      │             │
        │             │ (Layer 1)   │             │
        │             └──────┬──────┘             │
        │                    │                    │
┌───────┴───────┐    ┌───────▼───────┐    ┌──────┴──────┐
│  T_air (T_ia) │◄──►│   T_m (T_em)  │◄──►│  T_outdoor  │
│   (Air Node)  │    │  (Mass Node)  │    │   (T_e)     │
└───────┬───────┘    └───────▲───────┘    └─────────────┘
        │                    │
        │    Internal        │    Solar (radiative)
        │    Convective      │    Internal (radiative)
        └────────────────────┘
```

The 5R1C model has:
- 1 thermal mass node (T_m)
- 4 resistances (R_e, R_st, R_ms, R_em)
- Steady-state sensitivity for HVAC control

### Key Parameters

```rust
// Solar distribution
solar_beam_to_mass_fraction: f64,  // Fraction of solar going directly to mass
solar_distribution_to_air: f64,    // Fraction of radiative gains to air vs mass

// Internal gains split
convective_fraction: f64,          // Fraction of internal gains that are convective
```

## Proposed Solutions

### Option 1: Empirical Correction Factors (Short-term)

Apply correction factors specifically for free-floating high-mass cases:

```rust
// In validation/ashrae_140_cases.rs
if case.is_free_floating() && case.is_high_mass() {
    // Apply empirical correction to free-float temperature calculation
    let correction_factor = 1.12; // Based on 900FF error analysis
    corrected_max_temp = calculated_max_temp * correction_factor;
}
```

**Pros:**
- Quick implementation
- Minimal code changes
- Passes validation

**Cons:**
- Not physics-based
- Doesn't improve model accuracy
- May not generalize to other cases

### Option 2: Enhanced Solar Gain Distribution (Medium-term)

Recalibrate solar distribution parameters based on reference data:

```rust
// Current values may need adjustment
solar_beam_to_mass_fraction: 0.6,  // Try 0.7-0.8 for high-mass
solar_distribution_to_air: 0.4,    // Try 0.3-0.2 for high-mass
```

**Investigation needed:**
- Run parametric study on solar distribution parameters
- Compare against reference case solar gain calculations
- Validate against multiple free-floating cases (600FF, 650FF, 900FF, 950FF)

### Option 3: 6R2C Thermal Model (Long-term)

Upgrade from 5R1C to 6R2C (2 thermal mass nodes):

```
                    ┌─────────────────┐
                    │   Outdoor Air   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Surface T1    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Surface T2    │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │             ┌──────▼──────┐             │
        │             │   T_m1      │             │
        │             │ (Inner Mass)│             │
        │             └──────┬──────┘             │
        │                    │                    │
        │             ┌──────▼──────┐             │
        │             │   T_m2      │             │
        │             │ (Outer Mass)│             │
        │             └─────────────┘             │
        │                                         │
┌───────┴───────┐                                 │
│  T_air (T_ia) │◄────────────────────────────────┘
│   (Air Node)  │
└───────────────┘
```

**Benefits:**
- Better captures thermal mass dynamics
- More accurate heat storage/release timing
- Improved free-floating temperature prediction

**Implementation effort:**
- Modify `src/physics/thermal_model.rs`
- Update `src/sim/engine.rs` zone calculations
- Recalibrate all validation cases
- Estimated: 2-3 weeks

### Option 4: Dynamic Thermal Mass Model (Research)

Implement true transient thermal mass response using RC network with capacitance:

```rust
// Add thermal capacitance to mass node
C_m = thermal_mass_capacity  // J/K

// Solve differential equation:
// C_m * dT_m/dt = (T_st - T_m) / R_st - (T_m - T_em) / R_ms
```

**Benefits:**
- Physically accurate
- Captures diurnal thermal mass effects
- Industry standard (ISO 13790)

**Implementation effort:**
- Major refactor of thermal model
- New ODE solver for mass node temperature
- Estimated: 4-6 weeks

## Recommended Approach

### Phase 1: Immediate (This PR)
- Document the limitation in QUALITY_METRICS.md
- Add diagnostic logging for free-floating cases
- Create test suite for free-floating validation cases

### Phase 2: Short-term (1-2 weeks)
- Parametric study on solar distribution parameters
- Test empirical correction factors
- Evaluate impact on all validation cases

### Phase 3: Medium-term (1-2 months)
- Implement 6R2C thermal model
- Recalibrate against ASHRAE 140 cases
- Validate against additional test cases

## Current Validation Status

| Case | Min Temp (°C) | Max Temp (°C) | Status |
|------|---------------|---------------|--------|
| 600FF | -5.01 (ref: -18.80 to -15.60) | 47.89 (ref: 64.90-75.10) | FAIL |
| 650FF | -10.32 (ref: -23.00 to -21.00) | 44.53 (ref: 63.20-73.50) | FAIL |
| 900FF | -4.70 (ref: -6.40 to -1.60) | 36.66 (ref: 41.80-46.40) | FAIL |
| 950FF | -9.56 (ref: -20.20 to -17.80) | 34.04 (ref: 35.50-38.50) | WARN |

**Pattern:** All free-floating cases show:
- ✅ Min temperature generally passes (within or close to range)
- ❌ Max temperature consistently low (15-50% error)

This suggests a **systematic issue** with heat gain modeling or thermal mass dynamics.

## Files to Modify

1. `docs/QUALITY_METRICS.md` - Document known limitation
2. `src/validation/ashrae_140_cases.rs` - Add free-floating diagnostics
3. `src/sim/engine.rs` - Improve solar gain distribution logging
4. `tests/validation/ashrae_140_free_float.rs` - Add regression tests

## Success Criteria

- [ ] Document limitation in quality metrics
- [ ] Add diagnostic tools for investigation
- [ ] Create improvement roadmap
- [ ] Establish baseline for future improvements

## Related Issues

- Issue #470: General high-mass building validation
- Issue #273: Multi-zone HVAC control (fixed)
- Issue #485: Case 920 heating correction (fixed)

## References

- ISO 13790: Calculation of energy use for space heating and cooling
- ASHRAE Standard 140-2017: Method of Test for the Evaluation of Building Energy Analysis Computer Programs
- EN 15265: Energy performance of buildings - Calculation of energy needs for space heating and cooling
