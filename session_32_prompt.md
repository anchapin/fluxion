# Session 32 Prompt: Fix Case 960 and Deep Physics Issues

## Session Context

**Previous Session**: Session 31 - Partial success, peak power tracking fixed
**Current State**: 
- Pass Rate: 1.6% (1/64)
- Peak power: Now varies correctly ✅
- Case 960: 22.06 MWh cooling (FAIL - 6x over reference)
- Free-floating: Still failing

---

## Session 32 Task: Fix Case 960 Inter-Zone Coupling and Free-Floating Physics

### Priority 1: Fix Case 960 Cooling Overprediction

**Problem**: 
- Case 960 shows 22.06 MWh cooling vs 1.0-3.5 MWh reference (6x over!)
- This is the sunspace/buffer zone case with two zones:
  - Zone 0: Back-zone (conditioned)
  - Zone 1: Sunspace (buffer zone with south-facing windows)
- Inter-zone coupling might be causing massive heat transfer

**Investigation Steps**:
1. Check inter-zone conductance calculation in engine.rs around line 1542
2. Look at window area assignments for both zones - verify correct distribution
3. Check if COP correction (line 982-986 in validator) is actually being applied
4. Compare with 5R1C vs 6R2C model - which path is Case 960 using?
5. Look at hvac_equipment for Case 960 - is it using IdealHVACController?

**Root Cause Hypothesis**:
- Inter-zone coupling too high (1.5 W/K shown in debug output)
- Or COP correction not applied to annual_cooling_energy
- Or zone 1 (sunspace) is heating zone 0 excessively

### Priority 2: Fix Free-Floating Temperature Physics

**Problem**: 
- Cases 600FF, 650FF, 900FF, 950FF show incorrect temperatures
- Expected: Case 600FF min ~-18°C, max ~65°C
- Actual: Case 600FF min -6.70°C, max 38.88°C
- Too warm in summer, not cold enough in winter

**Investigation Steps**:
1. Check if solar gains are correctly zeroed for FF cases (line ~4887 in engine.rs)
2. Check if hvac_enabled flag is properly false for FF cases
3. Look at ground coupling - is it still adding heat?
4. Check thermal mass - is it buffering too much?
5. Compare with 5R1C vs CTF model - which is being used?

**Root Cause Hypothesis**:
- Solar gains still being applied (or not reduced enough)
- Ground coupling adding heat even without HVAC
- Thermal mass preventing temperature extremes

---

## Important Notes

1. **WORK INCREMENTALLY** - Test after each change
2. **USE RELEASE BUILD**: `RUST_MIN_STACK=16777216 cargo build --release`
3. **TEST AFTER EACH CHANGE**: Run validation to check impact
4. **USE DEBUG OUTPUT**: Check for debug prints in Case 960 validation
5. **CHECK BOTH PATHS**: Case 960 might use 5R1C or 6R2C

---

## Expected Outcome

1. Case 960 cooling reduced significantly (target: <10 MWh, ideal: <5 MWh)
2. Free-floating temperatures improved
3. Pass rate ≥14% (restore baseline)
4. No NEW empirical factors added

---

## Files to Investigate

- `src/sim/engine.rs`:
  - Inter-zone coupling (line ~1542)
  - Free-floating solar gains (line ~4887)
  - hvac_enabled flag
  - Ground coupling for FF cases
  
- `src/validation/ashrae_140_validator.rs`:
  - COP correction for Case 960 (line 982-986)
  - Energy calculation for Case 960

---

## Debug Commands

```bash
# Run specific case with debug output
cargo run --release --bin fluxion -- validate --case 960

# Check peak powers
cargo run --release --bin fluxion -- validate 2>&1 | grep "Peak"
```

---

## Success Criteria

- [ ] Case 960 cooling <10 MWh (was 22.06)
- [ ] Free-floating temperatures improved
- [ ] Pass rate ≥14%
- [ ] Peak power varies correctly (already fixed in Session 31)
- [ ] No NEW empirical factors added