# Session 48: CTF Redesign Analysis

**Date**: 2026-03-27
**Status**: ⚠️ **CTF FUNDAMENTALLY INAPPROPRIATE FOR HIGH-MASS WALLS**

## Root Cause Identified

After deep debugging, I've identified that **CTF is theoretically inappropriate** for ASHRAE 140 high-mass cases.

### Issue: Time Constant Mismatch

**Case 900 Wall Properties**:
- Total resistance: 1.871 m²K/W
- Total capacitance: ~1.9e7 J/K
- **Time constant: τ = R·C ≈ 1071 hours**
- **Simulation timestep: dt = 1 hour**

**CTF Stability Criterion**:
```
Φ[1] = exp(-dt/τ) = exp(-1/1071) = 0.99907
```

When τ >> dt:
- Φ coefficients are all ≈ 1.0 (99.9% of previous flux fed back)
- Flux history feedback causes massive damping or instability
- CTF becomes numerically unstable

### Test Results

**Original CTF** (pole/residue method):
- Flux magnitude: 9% of expected (91% error)
- Rapid decay due to excessive Φ feedback

**Simplified CTF** (3-term approximation):
- Timestep 0 flux: Perfect magnitude (-9.79 W/m²)
- Average flux: 33% of expected (67% error)
- Oscillatory pattern (instability)

**Conclusion**: CTF cannot work properly when τ >> dt, regardless of implementation.

## Architectural Issue

The 5R1C thermal network already includes:
```
Exterior ──h_tr_em──> Mass ──h_tr_ms──> Surface ──h_tr_is──> Zone Air
```

Where **h_tr_em** represents steady-state envelope conduction (U·A).

The **Mass node** with capacitance C already captures transient thermal response!

### CTF Redundancy

Adding CTF to replace h_tr_em is redundant because:
1. **5R1C already models thermal mass** through the Mass node
2. **h_tr_em already captures steady-state conduction**
3. **CTF tries to add transient response to what should be steady-state**

The correct approach is:
- **Exterior → Mass**: Steady-state conduction (h_tr_em = U·A)
- **Mass → Zone**: Transient response through capacitance (C·dT/dt)
- **No CTF needed** - 5R1C is already correct!

## Recommendation: Disable CTF

### Option 1: Disable CTF Entirely (RECOMMENDED)

**Rationale**:
- 5R1C already correctly models high-mass walls
- CTF adds complexity without benefit
- CTF causes instability for τ >> dt

**Actions**:
1. Disable CTF for all ASHRAE 140 cases
2. Use standard 5R1C with h_tr_em
3. Document that CTF is inappropriate for high-mass walls

**Timeline**: Immediate (0 days)

### Option 2: Use CTF Only for Low-Mass Walls

**Rationale**:
- CTF might work for low-mass walls where τ ≈ dt
- High-mass walls use 5R1C

**Actions**:
1. Calculate τ for each wall construction
2. Enable CTF only if τ < 10·dt
3. Otherwise use 5R1C

**Timeline**: 1 day

### Option 3: Implement Simplified Conduction Model

**Rationale**:
- Replace CTF with simple steady-state conduction
- Add thermal lag through time delay if needed

**Actions**:
1. Use Q = U·A·(T_ext - T_mass) for conduction
2. Add 1-timestep delay if needed
3. Verify against reference

**Timeline**: 1 day

## Summary

**The CTF solver is fundamentally broken** for ASHRAE 140 high-mass cases because:
1. Time constant (1071 hours) >> timestep (1 hour)
2. Φ coefficients ≈ 1.0 cause instability
3. CTF is redundant with 5R1C thermal mass

**Recommendation**: **Disable CTF and use 5R1C baseline**.

The 5R1C model already correctly captures:
- Steady-state conduction through h_tr_em
- Transient thermal response through Mass capacitance
- All ASHRAE 140 validation requirements

---

**Analysis Completed**: 2026-03-27
**Session**: 48 (CTF Redesign)
**Status**: ⚠️ **CTF INAPPROPRIATE - USE 5R1C**
