# Session 48: CTF Fix Testing Guide

**Date**: 2026-03-27
**Purpose**: Validate CTF flux integration fix

## Test Scripts Overview

I've created three test scripts to validate the CTF fix:

### 1. **Bash Script** (`test_ctf_fix.sh`)
Simple shell script that runs validation and parses output.

**Usage**:
```bash
./test_ctf_fix.sh
```

**Pros**:
- Simple, no dependencies
- Easy to read and modify
- Color-coded output

**Cons**:
- Basic parsing only
- Limited error handling

### 2. **Python Script** (`test_ctf_fix.py`) ⭐ **RECOMMENDED**
Comprehensive Python script with detailed validation.

**Usage**:
```bash
python test_ctf_fix.py
```

**Pros**:
- Comprehensive validation
- Detailed flux analysis
- Better error messages
- Easy to extend

**Cons**:
- Requires Python 3

### 3. **Rust Binary** (`src/bin/test_ctf_fix.rs`)
Full Rust implementation for deep debugging.

**Usage**:
```bash
cargo run --bin test_ctf_fix
```

**Pros**:
- Direct access to internals
- Can add custom diagnostics
- Type-safe

**Cons**:
- More complex to modify
- Slower compile-test cycle

## What the Tests Check

### ✅ Annual Energies
- **Reference Range** (ASHRAE 140):
  - Heating: 1.17 - 2.04 MWh
  - Cooling: 2.13 - 3.67 MWh
- **Expected**: PASS (both within range)

### ⚠️ Peak Loads
- **Reference Range** (ASHRAE 140):
  - Heating: 1.80 - 2.40 kW
  - Cooling: 1.60 - 2.10 kW
- **Expected**: WARN (may be out of range, that's OK)

### ✅ Flux Integration
- **Checks**:
  - CTF solver is active
  - SESSION 48 FIX debug output present
  - No "Q_net=" (old buggy version)
  - Flux magnitudes are reasonable (0.1 - 1000 W)
- **Expected**: PASS

## Interpreting Results

### Success Case ✅
```
✅ ALL CRITICAL TESTS PASSED

Annual Energies: ✅ PASS
Peak Loads: ✅ PASS (warnings only)
Flux Integration: ✅ PASS
```

**What it means**:
- CTF fix is working correctly
- Network topology is preserved
- Energy conservation maintained
- Ready for Session 49

### Failure Case ❌
```
❌ SOME TESTS FAILED

Annual Energies: ❌ FAIL
Peak Loads: ⚠️ WARN
Flux Integration: ❌ FAIL
```

**What to check**:
1. Is CTF enabled? (grep for "CTF solver ACTIVE")
2. Is fix applied? (grep for "SESSION 48 FIX")
3. Check flux values (grep for "Q_CTF=")
4. Review SESSION_48_CTF_FLUX_INTEGRATION_ISSUE.md

## Debugging Tips

### Check 1: CTF Enabled
```bash
cargo run --release --bin fluxion validate --case 900 2>&1 | grep "CTF solver"
```

**Expected output**:
```
✅ SESSION 48: CTF solver ACTIVE - using CTF for envelope conduction
🔧 SESSION 48 FIX: CTF flux to zone air: Q_CTF=XXX.XX W
```

### Check 2: No Old Buggy Code
```bash
cargo run --release --bin fluxion validate --case 900 2>&1 | grep "Q_net="
```

**Expected output**: (nothing - should not find "Q_net=")

### Check 3: Flux Magnitudes
```bash
cargo run --release --bin fluxion validate --case 900 2>&1 | grep "Q_CTF=" | head -3
```

**Expected output**: Flux values in range 0.1 - 1000 W

### Check 4: Network Topology
```bash
cargo run --release --bin fluxion validate --case 900 2>&1 | grep "h_ext"
```

**Expected**: No errors related to h_ext calculation

## Common Issues

### Issue 1: "CTF solver INACTIVE"
**Cause**: CTF not enabled for this case
**Fix**: Check `ctf_enabled` flag in code

### Issue 2: "Q_net=" in output
**Cause**: Old buggy version still running
**Fix**: Ensure SESSION 48 FIX is applied (check engine.rs lines 3632-3660)

### Issue 3: Flux magnitudes ~0 or >10000
**Cause**: Calculation error or wrong boundary conditions
**Fix**: Check CTF solver initialization and boundary temps

### Issue 4: Annual energies out of range
**Cause**: Energy conservation violation
**Fix**: Check energy balance calculation

## Quick Validation Commands

### Run all tests
```bash
# Python (recommended)
python test_ctf_fix.py

# Bash
./test_ctf_fix.sh

# Rust
cargo run --bin test_ctf_fix
```

### Check specific metrics
```bash
# Annual energies
cargo run --release --bin fluxion validate --case 900 | grep "Annual"

# Peak loads
cargo run --release --bin fluxion validate --case 900 | grep "Peak"

# CTF flux
cargo run --release --bin fluxion validate --case 900 | grep "Q_CTF"
```

## Success Criteria

The fix is successful if:

1. ✅ **Annual energies pass** (within ASHRAE 140 range)
2. ✅ **No "Q_net=" in output** (old buggy code removed)
3. ✅ **Flux magnitudes reasonable** (0.1 - 1000 W)
4. ✅ **Energy conservation < 1%** (no major imbalances)
5. ⚠️ **Peak loads may warn** (that's OK, not critical)

## Next Steps After Success

1. **Document results** in SESSION_48_RESULTS.md
2. **Run full year validation** for all 900-series cases
3. **Compare with 5R1C baseline** to ensure improvements
4. **Proceed to Session 49** with CTF enabled

## Contact

If tests fail:
1. Check SESSION_48_CTF_FLUX_INTEGRATION_ISSUE.md for root cause
2. Check SESSION_48_CTF_FIX_IMPLEMENTATION.md for implementation details
3. Review git diff to ensure all changes are applied
4. Run with debug output: `RUST_LOG=debug cargo run --release --bin fluxion validate --case 900`

---

**Test Scripts Created**: 2026-03-27
**Session**: 48 (CTF Solver Audit - Fix Validation)
**Status**: Ready for testing
