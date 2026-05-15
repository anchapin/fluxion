# Session 31 Prompt: Restore Baseline & Fix Critical Physics Bugs

## Project Context
You are working on **Fluxion**, a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. The project validates against **ASHRAE Standard 140** (thermal modeling validation suite).

## Session 30 Recap
- **Objective**: Fix cooling predictions and improve pass rate
- **Result**: FAILED - Project in degraded state (3.1% pass rate vs 14.1% baseline)
- **Cause**: Editing validator caused structural damage; git restore required

## Current Baseline Issues (After Git Restore)

### Critical Failures:
1. **Free-floating temperatures BROKEN**:
   - Case 600FF shows -5°C min vs -18.8°C reference
   - Case 650FF shows -10°C vs -23°C reference
   - Case 900FF shows -0.7°C vs -6.4°C reference

2. **Case 960 Catastrophic Failure**:
   - Cooling: 22.06 MWh vs 1.00-3.50 reference (6x OVER)
   - Heating: 0.06 MWh vs 5.00-15.00 reference (WAY UNDER)
   - This is the sunspace case - inter-zone coupling broken

3. **Peak Power Tracking BROKEN**:
   - All 600/900-series show exactly 2.10 kW peak heating
   - No variation - indicates tracking not working

4. **600-series heating/cooling wrong**:
   - Case 600: 6.79/6.53 vs 5.50-7.50/8.00-10.50 reference

### Only Passing Cases:
- Case 610 cooling (4.56 vs 3.92-6.14)
- Case 640 cooling (6.40 vs 5.95-8.10)
- Case 920 heating/cooling (both pass)
- Case 930 heating/cooling (both pass)
- Case 950 cooling (0.95 vs 0.39-0.92)

**Pass Rate: 3.1% (2/64)**

---

## Session 31 Task: Fix Critical Physics Bugs

### Priority 1: Fix Free-Floating Temperature Failure

**Problem**: Cases 600FF, 650FF, 900FF showing wrong temperatures
- Expected: Case 600FF min temp ~-18°C, max ~65°C
- Actual: Case 600FF min -5°C, max 48°C

**Investigation Steps**:
1. Check `hvac_enabled` flag for FF cases in `engine.rs`
2. Verify solar gains are NOT applied to free-floating cases
3. Check internal loads NOT applied to FF cases
4. Verify ground coupling is working correctly
5. Look at how timestep calculation handles no-HVAC

**Root Cause Hypothesis**:
- FF cases might still be getting solar gains or internal loads
- Or ground coupling might be applying incorrectly

### Priority 2: Fix Case 960 Cooling Overprediction

**Problem**: 22 MWh vs 1-3.5 MWh reference (6x)
- This is the sunspace/buffer zone case
- Two-zone model: Zone 0 (sunspace) + Zone 1 (back-zone)

**Investigation Steps**:
1. Check if COP correction (line 982) is double-counted
2. Check inter-zone coupling conductance
3. Verify sunspace heat transfer to back-zone
4. Look at how zone 0 (sunspace) cooling is calculated

**Root Cause Hypothesis**:
- COP division happening twice
- Or inter-zone coupling too high causing thermal runaway

### Priority 3: Fix Peak Power Tracking

**Problem**: All cases show exactly 2.10 kW peak heating
- Should vary by case (2.8-6.1 kW for 600-series)

**Investigation Steps**:
1. Check peak_power_heating tracking in engine
2. Verify HVAC demand calculation per timestep
3. Check if setpoint differences are being captured

### Priority 4: Document Current Empirical Corrections

Before making ANY changes, document what's in validator:
```bash
grep -n "correction\|factor\|multiply\|divide" src/validation/ashrae_140_validator.rs | head -50
```

**Note on Directory Viewing**: When exploring the codebase or viewing large directories, use `ls -la` instead of plain `ls` to see file details and permissions incrementally.

---

## Important Notes

1. **NO NEW EMPIRICAL CORRECTIONS** - Fix root causes in physics engine
2. **Work INCREMENTALLY** - Test after each change
3. **USE RELEASE BUILD**: `RUST_MIN_STACK=16777216 cargo build --release`
4. **TEST AFTER EACH CHANGE**: Run validation to check impact
5. **DOCUMENT** any changes in SESSION_31_SUMMARY.md

---

## Expected Outcome
1. Pass rate restored to ≥14% (Session 29 baseline)
2. Free-floating temperatures working for FF cases
3. Case 960 cooling within reasonable range (not 22 MWh)
4. Peak power shows variation between cases
5. No NEW empirical factors added

---

## Files to Investigate
- `src/sim/engine.rs` - Physics calculations, hvac_enabled, peak tracking
- `src/validation/ashrae_140_validator.rs` - Current corrections (read only)
- Look for: hvac_enabled, free_floating, peak_power, solar_gains handling

---

## Success Criteria
- [ ] Free-floating cases (600FF, 650FF, 900FF) within reference
- [ ] Case 960 cooling <10 MWh (was 22)
- [ ] Peak power varies by case (not all 2.10)
- [ ] Pass rate ≥14%
- [ ] NO new empirical factors added
