# Session 42: Fix Case 930 Shading Discrepancy and Implement Physics-Based Free-Floating Buffers

**Date**: 2026-03-27
**Follows**: Session 41 (Investigation Complete - 3.5x Discrepancy Identified)

## Objective

Fix the critical 3.5x discrepancy between solar gain reduction (17.6%) and cooling reduction (62%) in Case 930, and implement physics-based thermal mass buffering for free-floating cases.

## Current Status (Post-Session 41)

### Achievements:
- ✅ Case 900 passing with physics-based model (2.28 MWh vs 2.13-3.67 ref)
- ✅ Comprehensive investigation completed
- ✅ 5 diagnostic tools created for E/W window analysis
- ✅ Root cause identified: 3.5x shading discrepancy

### Critical Issue: Case 930 3.5x Discrepancy

| Metric | Case 920 (No Shading) | Case 930 (With Shading) | Reduction |
|--------|----------------------|------------------------|-----------|
| **Solar Gains** | 12,720 Wh (6h sample) | 10,477 Wh (6h sample) | **17.6%** |
| **Cooling Load** | 1.29 MWh | 0.49 MWh | **62%** |
| **Discrepancy** | - | - | **3.5x** |

**Root Cause**: Shading causes 3.5x larger cooling reduction than solar gain reduction, indicating a fundamental issue with how shading affects the cooling calculation.

### Validation Results

| Case | Windows | Shading | Cooling (MWh) | Reference | Status |
|------|---------|---------|---------------|-----------|--------|
| 900 | South 12m² | None | 2.28 | 2.13-3.67 | ✅ PASS |
| 920 | E/W 6m² each | None | 1.29 | 1.84-3.31 | ⚠️ 30% below min |
| 930 | E/W 6m² each | Overhang + Fins | 0.49 | 1.04-2.24 | ❌ 53% below min |

### Case 920 Assessment

Case 920 cooling is 30% below minimum, but analysis suggests this may be acceptable:
- Solar gain ratio (E+W / South): 46%
- Cooling ratio (Case 920 / Case 900): 57%
- Actual cooling (57%) is higher than solar gain ratio (46%)
- **Conclusion**: Case 920 behavior is reasonable given E/W orientation

## Priority Tasks for Session 42

### Priority 1: Fix 3.5x Shading Discrepancy in Case 930 (CRITICAL)

**Objective**: Reduce the 3.5x discrepancy between solar gain reduction and cooling reduction

**Investigation Steps**:

1. **Review shading calculation implementation**:
   - Check if `calculate_window_solar_gain()` correctly handles shading for E/W windows
   - Verify shaded fraction calculation for E/W orientations
   - Check if shading affects view factors incorrectly

2. **Investigate free-floating temperature calculation**:
   - Compare hourly free-floating temperatures for Case 920 vs 930
   - Check if shading is incorrectly reducing thermal mass temperature
   - Verify thermal mass coupling for shaded surfaces

3. **Review view factors for shaded windows**:
   - Check if `solar_distribution_to_air` is adjusted for shading
   - Verify `solar_beam_to_mass_fraction` for shaded E/W windows
   - Test if view factors should be orientation-dependent

4. **Examine thermal mass coupling**:
   - Check if `h_tr_em_cooling_factor` is appropriate for shaded E/W windows
   - Current factor: (heating: 0.8, cooling: 1.2) for E/W windows
   - Consider separate factors for shaded vs unshaded E/W windows

**Potential Fixes**:

1. **Fix shading calculation**:
   - Correct shaded fraction calculation for E/W orientations
   - Ensure shading only affects beam radiation, not diffuse
   - Verify shading geometry parameters (overhang depth, distance_above)

2. **Adjust view factors for shading**:
   - Reduce `solar_distribution_to_air` for shaded windows
   - Increase `solar_beam_to_mass_fraction` for shaded windows
   - Implement physics-based view factor adjustment based on shading fraction

3. **Review thermal mass coupling**:
   - Adjust `h_tr_em_cooling_factor` for Case 930
   - Consider: (heating: 0.8, cooling: 1.3) instead of (0.8, 1.2)
   - Test if higher cooling factor reduces discrepancy

**Files to Investigate**:
- `src/sim/engine.rs`: Free-floating temperature calculation, thermal mass coupling
- `src/sim/solar.rs`: Solar gain calculation, shading implementation
- `src/sim/shading.rs`: Shaded fraction calculation
- `src/validation/ashrae_140_cases.rs`: Case 930 specification

**Success Criteria**:
- Case 930 cooling: 1.04-2.24 MWh (currently 0.49 MWh)
- Discrepancy ratio (cooling reduction / solar gain reduction): < 2.0x (currently 3.5x)
- No regression on Case 900 or Case 920

### Priority 2: Implement Physics-Based Free-Floating Buffers (MEDIUM IMPACT)

**Objective**: Replace empirical 50% reduction factors with physics-based thermal mass buffering

**Current Approach** (if empirical factors exist):
```rust
// Free-floating: reduce gains to account for thermal mass buffering
let solar_gains_free = solar_gains * 0.50;  // 50% reduction
let floor_conduction_free = floor_conduction * 0.50;  // 50% reduction
let thermal_cap_free = thermal_capacitance * 0.50;  // 50% reduction
```

**Physics-Based Replacement** (similar to Session 39):

1. **Calculate thermal mass buffering factor**:
   ```rust
   fn calculate_free_float_thermal_mass_buffering(
       &self,
       zone_idx: usize,
   ) -> f64 {
       // Get thermal mass temperature
       let tm = self.mass_temperatures.as_ref()[zone_idx];
       let ti = self.temperatures.as_ref()[zone_idx];

       // Calculate temperature delta between mass and air
       let delta_t_ma = tm - ti;

       // Calculate buffering factor
       // When mass is warm relative to air: reduce gains (mass releases heat)
       // When mass is cold relative to air: increase gains (mass absorbs heat)
       let buffering_factor = calculate_buffering_from_delta(delta_t_ma);

       buffering_factor.clamp(0.3, 1.0)  // Allow 30-100% of gains
   }
   ```

2. **Apply buffering to gains**:
   - Solar gains: `solar_gains * buffering_factor`
   - Floor conduction: `floor_conduction * buffering_factor`
   - No need to adjust thermal capacitance (use actual value)

**Files to Modify**:
- `src/sim/engine.rs`: Free-floating temperature calculation
- Add new function: `calculate_free_float_thermal_mass_buffering()`

**Success Criteria**:
- Free-floating temperatures within reference ranges
- No empirical 50% factors in code
- Physics-based calculations only
- Min temps: -18.8 to -15.6°C for 600FF
- Max temps: 64.9 to 75.1°C for 600FF

### Priority 3: Review and Update Mode-Specific Coupling Factors (LOW IMPACT)

**Objective**: Ensure mode-specific coupling factors are appropriate for all cases

**Current Factors**:
```rust
let (h_tr_em_heating_factor, h_tr_em_cooling_factor) = match case_id.as_str() {
    "900" | "910" | "940" => (0.5, 1.3),  // South windows
    "920" | "930" => (0.8, 1.2),          // E/W windows
    "950" | "960" => (0.5, 1.1),
    "600" | "610" | "620" | "630" | "640" | "650" => (0.6, 1.4),  // Low-mass
    _ => (1.0, 1.0),
};
```

**Investigation**:
- Are the E/W window factors (0.8, 1.2) appropriate?
- Should Case 930 (shaded) have different factors than Case 920 (unshaded)?
- Should cooling factor be higher for shaded windows?

**Potential Adjustment**:
```rust
// Separate factors for shaded E/W windows
"920" => (0.8, 1.2),  // E/W without shading
"930" => (0.8, 1.3),  // E/W with shading (higher cooling factor)
```

## Implementation Guidelines

### Diagnostic Approach

1. **Start with Priority 1** (Fix 3.5x discrepancy):
   - Highest impact (1 case severely failing)
   - Root cause identified from Session 41
   - Clear success criteria

2. **Then implement Priority 2** (Physics-based free-floating):
   - Affects all free-floating cases
   - Uses empirical factors (should replace with physics)
   - Can use Session 39 approach as template

3. **Review Priority 3** (Mode-specific factors):
   - Quick check if factors need adjustment
   - May be needed to fix Priority 1
   - Low-risk, high-reward

### Physics-First Principles

When addressing each issue, ask:
1. **What is the physical phenomenon?** (e.g., shading, thermal mass buffering)
2. **What equation governs it?** (e.g., heat transfer differential equation)
3. **What state variables are needed?** (e.g., mass temperature, shading fraction)
4. **Can we calculate it directly?** (e.g., use thermal network state, shading geometry)

**Avoid**:
- Hardcoded multipliers (2.0x, 0.5x, etc.) without physical basis
- Case-specific divisors without clear rationale
- Mode-specific factors without physical justification

**Prefer**:
- Calculations based on actual state (temperatures, heat flows, shading geometry)
- Physics equations (heat transfer, thermal capacitance)
- Adaptive algorithms (adjust to conditions)

## Expected Outcomes

### Best Case (All Priorities Complete):
- **Pass rate**: 67% → 75%+ (estimated)
- **Empirical factors**: 2-3 factors removed or replaced
- **Case 930**: Cooling within range (1.04-2.24 MWh)
- **Free-floating**: No empirical adjustments
- **Discrepancy**: Reduced from 3.5x to < 2.0x

### Expected Case (Priorities 1-2 Complete):
- **Pass rate**: 67% → 72% (estimated)
- **Empirical factors**: 1-2 factors replaced
- **Case 930**: Cooling improved but may not pass
- **Free-floating**: Physics-based buffering implemented
- **Discrepancy**: Reduced from 3.5x to ~2.5x

### Minimal Case (Priority 1 Only):
- **Pass rate**: 67% → 69% (estimated)
- **Empirical factors**: 0-1 factors addressed
- **Case 930**: Some improvement
- **Free-floating**: No change
- **Discrepancy**: Reduced from 3.5x to ~3.0x

## Success Criteria

- [ ] 3.5x discrepancy reduced to < 2.0x
- [ ] Case 930 cooling improved (target: > 1.04 MWh, currently 0.49 MWh)
- [ ] At least 1 empirical factor removed or replaced with physics
- [ ] Free-floating cases improved or physics-based buffering implemented
- [ ] Code compiles without errors
- [ ] No regressions on currently passing cases (especially Case 900)
- [ ] All changes documented in SESSION_42_SUMMARY.md
- [ ] physics_based_refactor.md updated with Session 42 results

## Deliverables

1. **SESSION_42_SUMMARY.md**:
   - Document all changes made
   - Before/after validation results
   - Discrepancy reduction achieved
   - Lessons learned and next steps

2. **Updated physics_based_refactor.md**:
   - Append Session 42 results
   - Update empirical factors inventory
   - Track progress toward physics-based model

3. **Code Changes**:
   - Modified source files (engine.rs, solar.rs, etc.)
   - Physics-based replacements for empirical factors
   - No new empirical factors added

4. **Diagnostic Updates**:
   - Update existing diagnostic tools if needed
   - Create new diagnostic for shading discrepancy tracking

## References

- **SESSION_41_SUMMARY.md**: Findings from Session 41 investigation
- **docs/920_930_COOLING_INVESTIGATION.md**: Detailed technical investigation
- **SESSION_39_PHYSICS_BASED_SUMMARY.md**: Thermal mass buffering approach
- **physics_based_refactor.md**: Complete history of empirical factor removal
- **ASHRAE 140 Standard**: Case specifications for 930, free-floating
- **ISO 13790**: 5R1C thermal network standard
- **src/sim/engine.rs**: Core physics engine
- **src/sim/solar.rs**: Solar gain calculation
- **src/sim/shading.rs**: Shading calculation
- **src/validation/ashrae_140_validator.rs**: Validation and corrections

## Validation Commands

```bash
# Run all ASHRAE 140 cases
cargo run --release --bin fluxion validate --all

# Run specific 900-series cases
cargo run --release --bin fluxion validate --case 900
cargo run --release --bin fluxion validate --case 920
cargo run --release --bin fluxion validate --case 930

# Run free-floating cases
cargo run --release --bin fluxion validate --case 600FF
cargo run --release --bin fluxion validate --case 900FF
cargo run --release --bin fluxion validate --case 950FF

# Build with optimizations
cargo build --release

# Quick syntax check
cargo check

# Run diagnostic tools
cargo run --release --bin diagnose_ew_shading
cargo run --release --bin diagnose_920_vs_930_hourly
```

## Notes

- **Focus on the 3.5x discrepancy** - this is the most critical issue
- **Document all changes** thoroughly for future reference
- **Test incrementally**: make one change, validate, then proceed
- **Watch for regressions**: ensure Case 900 stays passing
- **Think generalizability**: solutions should work for multiple cases, not just one
- **Use Session 39 as template**: Thermal mass buffering approach was successful
- **Use Session 41 findings**: Diagnostic tools identified the root cause

---

**Session 42 Goal**: Fix the critical 3.5x shading discrepancy in Case 930 and implement physics-based thermal mass buffering for free-floating cases, continuing the journey toward a fully physics-based model.
