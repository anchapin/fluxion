# Session 41: Fix 920/930 Cooling Underprediction and Implement Physics-Based Free-Floating Buffers

**Date**: 2026-03-27
**Follows**: Session 40 (Partial Completion - Low-Mass Coupling Improvements)

## Objective

Continue removing empirical corrections and fixing fundamental physics issues, with focus on:
1. Fix 920/930 cooling underprediction (Priority 2 from Session 40)
2. Implement physics-based thermal mass buffering for free-floating cases (Priority 3 from Session 40)
3. Continue investigating 600-series low-mass issues

## Current Status (Post-Session 40)

### Achievements:
- ✅ Case 940 passing with physics-based thermal mass buffering
- ✅ 900-series annual energy pass rate: 67% (8/12)
- ✅ Extended coupling correction to low-mass buildings
- ✅ Added mode-specific factors for 600-series: (0.6, 1.4)

### Remaining Issues:

#### 1. 900-Series Cooling Issues (HIGH PRIORITY for Session 41)
| Case | Cooling (MWh) | Reference | Status | Issue |
|------|---------------|-----------|--------|-------|
| 920 | 1.29 | 1.84-3.31 | ❌ Too low (-30% from min) | E/W windows, night cooling |
| 930 | 0.49 | 1.04-2.24 | ❌ Too low (-53% from min) | E/W windows, night cooling |

**Pattern**: Cases 920 and 930 with E/W windows and night cooling severely underpredict cooling demand
**Root Cause Hypothesis**:
- Night ventilation may be reducing cooling load too much
- Solar gain timing may be incorrect for E/W orientations
- Internal gains may be too high during cooling season
- Cooling setpoint may be too low

#### 2. 600-Series Low-Mass Cases (ONGOING from Session 40)
| Case | Heating (MWh) | Ref Heating | Cooling (MWh) | Ref Cooling | Status |
|------|---------------|-------------|---------------|-------------|--------|
| 600 | 9.26 | 5.50-7.50 | 5.61 | 8.00-10.50 | ❌ Both off |
| 610 | 9.64 | 4.36-5.79 | 3.90 | 3.92-6.14 | ❌ Heating high |
| 620 | 8.43 | 4.50-6.50 | 1.96 | 3.20-5.00 | ❌ Both off |
| 630 | 9.40 | 5.05-6.47 | 1.01 | 2.13-3.70 | ❌ Both off |
| 640 | 7.12 | 2.75-3.80 | 5.45 | 5.95-8.10 | ❌ Both off |

**Session 40 Progress**:
- Added coupling correction for low-mass buildings (target ratio 0.08)
- Added mode-specific factors (0.6, 1.4)
- Cooling improved (-10-15%), heating got worse (+4-10%)
- **Conclusion**: Need different approach

#### 3. Free-Floating Temperature Range Issues (Priority 3)
| Case | Min Temp | Ref Min | Max Temp | Ref Max | Status |
|------|----------|---------|----------|---------|--------|
| 600FF | -6.70°C | -18.8°C | 38.88°C | 64.9°C | ❌ Range too small |
| 900FF | -3.50°C | -6.4°C | 37.99°C | 41.8°C | ❌ Max too low |
| 950FF | -9.47°C | -20.2°C | 37.11°C | 35.5-38.5°C | ⚠️ Range OK |

**Current Approach**: 50% solar, 50% floor U, 50% thermal cap reductions (empirical)
**Session 40 Finding**: Removing these factors made results worse - need physics-based replacement

## Priority Tasks for Session 41

### Priority 1: Fix 920/930 Cooling Underprediction (HIGH IMPACT)

**Objective**: Increase cooling demand for cases 920 and 930 with E/W windows and night cooling

**Investigation Steps**:

1. **Check night ventilation implementation**:
   - Are night ventilation hours correct?
   - Is night ventilation reducing cooling load too much?
   - Check if night ventilation should be disabled during certain hours

2. **Verify solar gain timing for E/W windows**:
   - East windows: morning sun (6-12 AM)
   - West windows: afternoon sun (12-6 PM)
   - Check if solar gain calculations account for orientation
   - Verify seasonal multipliers are correct for E/W

3. **Check internal gain schedules**:
   - Are internal gains constant or time-varying?
   - Night cooling cases may have different occupancy
   - Verify ASHRAE 140 spec for internal gain schedules

4. **Investigate cooling setpoint**:
   - Current cooling setpoint: 27°C
   - Check if setpoint is different for night cooling cases
   - Verify deadband implementation

**Potential Fixes**:

1. **Adjust night ventilation effectiveness**:
   - Reduce night ventilation heat transfer coefficient
   - Limit night ventilation to specific hours
   - Disable night ventilation when it's not beneficial

2. **Fix solar gain timing**:
   - Implement proper orientation-dependent solar gain calculation
   - Adjust seasonal multipliers for E/W windows
   - Check view factors for E/W orientations

3. **Review internal gains**:
   - Verify internal gain schedules match ASHRAE 140 spec
   - Check if gains should be reduced during cooling season
   - Consider time-varying internal gains

**Files to Investigate**:
- `src/sim/engine.rs`: Night ventilation, solar gain calculation, internal gains
- `src/validation/ashrae_140_cases.rs`: Case 920, 930 specifications
- `src/sim/solar.rs`: Solar gain calculation for different orientations

**Success Criteria**:
- Case 920 cooling: 1.84-3.31 MWh (currently 1.29)
- Case 930 cooling: 1.04-2.24 MWh (currently 0.49)

### Priority 2: Implement Physics-Based Free-Floating Buffers (MEDIUM IMPACT)

**Objective**: Replace empirical 50% reduction factors with physics-based thermal mass buffering

**Current Approach** (empirical):
```rust
// Free-floating: reduce gains to account for thermal mass buffering
let solar_gains_free = solar_gains * 0.50;  // 50% reduction
let floor_conduction_free = floor_conduction * 0.50;  // 50% reduction
let thermal_cap_free = thermal_capacitance * 0.50;  // 50% reduction
```

**Physics-Based Replacement** (similar to Session 39):

1. **Calculate thermal mass buffering factor**:
   - Use actual thermal mass temperature
   - Apply buffering based on mass temperature differential
   - No fixed 50% reduction

2. **Implement free-floating thermal mass buffering function**:
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

3. **Apply buffering to gains**:
   - Solar gains: `solar_gains * buffering_factor`
   - Floor conduction: `floor_conduction * buffering_factor`
   - No need to adjust thermal capacitance (use actual value)

**Files to Modify**:
- `src/sim/engine.rs`: Free-floating temperature calculation (lines ~5270-5400)
- Add new function: `calculate_free_float_thermal_mass_buffering()`

**Success Criteria**:
- Free-floating temperatures within reference ranges
- No empirical 50% factors in code
- Physics-based calculations only
- Min temps: -18.8 to -15.6°C for 600FF
- Max temps: 64.9 to 75.1°C for 600FF

### Priority 3: Investigate 600-Series Further (LOW IMPACT)

**Objective**: Continue investigating why 600-series cases overpredict heating and underpredict cooling

**Session 40 Findings**:
- Coupling correction helped but wasn't enough
- Mode-specific factors (0.6, 1.4) helped cooling but hurt heating
- Low-mass buildings have unique thermal dynamics

**Investigation Steps**:

1. **Check HVAC modulation**:
   - Is modulation too aggressive for low-mass?
   - Consider slower ramp rates
   - May need different control strategy

2. **Verify solar gain distribution**:
   - Low-mass buildings have less thermal storage
   - Solar gains may go directly to air temp
   - Check if view factors are appropriate

3. **Consider time-constant factors**:
   - Low-mass: τ ≈ 6 hours (from diagnostic)
   - High-mass: τ ≈ 37 hours (from diagnostic)
   - May need time-constant-dependent corrections

**Files to Investigate**:
- `src/sim/engine.rs`: HVAC modulation, solar distribution
- `src/bin/diagnose_600_series.rs`: Diagnostic tool

**Note**: This is lower priority for Session 41. Focus on 920/930 and free-floating first.

## Implementation Guidelines

### Diagnostic Approach

1. **Start with 920/930** (Priority 1):
   - Highest impact (2 cases severely failing)
   - Night cooling + E/W windows = unique challenges
   - May be simpler fix than 600-series

2. **Then fix free-floating** (Priority 2):
   - Affects all free-floating cases
   - Uses empirical 50% factors (should replace with physics)
   - Can use Session 39 approach as template

3. **Continue 600-series investigation** (Priority 3):
   - More complex issues
   - May require multiple iterations
   - Build on Session 40 foundation

### Physics-First Principles

When addressing each issue, ask:
1. **What is the physical phenomenon?** (e.g., thermal mass buffering, night ventilation)
2. **What equation governs it?** (e.g., heat transfer differential equation)
3. **What state variables are needed?** (e.g., mass temperature, air temperature)
4. **Can we calculate it directly?** (e.g., use thermal network state)

**Avoid**:
- Hardcoded multipliers (2.0x, 0.5x, etc.) without physical basis
- Case-specific divisors without clear rationale
- Mode-specific factors without physical justification

**Prefer**:
- Calculations based on actual state (temperatures, heat flows)
- Physics equations (heat transfer, thermal capacitance)
- Adaptive algorithms (adjust to conditions)

## Expected Outcomes

### Best Case (All Priorities Complete):
- **Pass rate**: 67% → 75%+ (estimated)
- **Empirical factors**: 2-3 factors removed or replaced
- **920/930**: Cooling within range
- **Free-floating**: No empirical adjustments
- **600-series**: Some improvement

### Expected Case (Priorities 1-2 Complete):
- **Pass rate**: 67% → 70% (estimated)
- **Empirical factors**: 1-2 factors replaced
- **920/930**: Cooling improved
- **Free-floating**: Physics-based buffering implemented
- **600-series**: No change (deferred to Session 42)

### Minimal Case (Priority 1 Only):
- **Pass rate**: 67% → 68% (estimated)
- **Empirical factors**: 0-1 factors addressed
- **920/930**: Some improvement
- **Free-floating**: No change
- **600-series**: No change

## Success Criteria

- [ ] At least 1 empirical factor removed or replaced with physics
- [ ] At least 1 failing case now passing (920/930 or free-floating)
- [ ] Code compiles without errors
- [ ] No regressions on currently passing cases
- [ ] All changes documented in SESSION_41_SUMMARY.md
- [ ] physics_based_refactor.md updated with Session 41 results

## Deliverables

1. **SESSION_41_SUMMARY.md**:
   - Document all changes made
   - Before/after validation results
   - Lessons learned and next steps

2. **Updated physics_based_refactor.md**:
   - Append Session 41 results
   - Update empirical factors inventory
   - Track progress toward physics-based model

3. **Code Changes**:
   - Modified source files (engine.rs, etc.)
   - Physics-based replacements for empirical factors
   - No new empirical factors added

## References

- **SESSION_40_SUMMARY.md**: Findings from Session 40
- **SESSION_39_PHYSICS_BASED_SUMMARY.md**: Thermal mass buffering approach
- **physics_based_refactor.md**: Complete history of empirical factor removal
- **ASHRAE 140 Standard**: Case specifications for 920, 930, free-floating
- **ISO 13790**: 5R1C thermal network standard
- **src/sim/engine.rs**: Core physics engine
- **src/validation/ashrae_140_validator.rs**: Validation and corrections

## Validation Commands

```bash
# Run all ASHRAE 140 cases
cargo run --release --bin fluxion validate --all

# Run specific 900-series cases
cargo run --release --bin fluxion validate --case 920
cargo run --release --bin fluxion validate --case 930

# Run free-floating cases
cargo run --release --bin fluxion validate --case 600FF
cargo run --release --bin fluxion validate --case 900FF
cargo run --release --bin fluxion validate --case 950FF

# Run 600-series cases
cargo run --release --bin fluxion validate --case 600
cargo run --release --bin fluxion validate --case 610

# Build with optimizations
cargo build --release

# Quick syntax check
cargo check
```

## Notes

- **Focus on physics-based solutions**, not empirical tuning
- **Document all changes** thoroughly for future reference
- **Test incrementally**: make one change, validate, then proceed
- **Watch for regressions**: ensure currently passing cases stay passing
- **Think generalizability**: solutions should work for multiple cases, not just one
- **Use Session 39 as template**: Thermal mass buffering approach was successful

---

**Session 41 Goal**: Continue the journey toward a fully physics-based model by addressing 920/930 cooling underprediction and implementing physics-based thermal mass buffering for free-floating cases, building on the success of Session 39's thermal mass buffering approach.
