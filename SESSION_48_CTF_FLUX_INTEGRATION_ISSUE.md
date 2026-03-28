# Session 48: CTF Flux Integration Issue - Root Cause Analysis

**Date**: 2026-03-27
**Status**: 🔴 CRITICAL INTEGRATION BUG IDENTIFIED
**Impact**: 12x flux magnitude mismatch causing peak load failures

## Problem Statement

CTF flux is **12x smaller** than 5R1C flux, causing significant peak load errors:
- CTF: -278.59 W
- 5R1C: -3270.51 W
- Net correction: +2991.92 W (wrongly added to mass balance)

## Root Cause Analysis

### The 5R1C Network Topology

The ISO 13790 5R1C network has this structure:

```
Exterior ──h_tr_em──> Mass ──h_tr_ms──> Surface ──h_tr_is──> Zone Air
                ↑                                ↑
            h_tr_w + h_ve (windows + ventilation)
```

**Key conductances**:
- `h_tr_em`: Exterior → Mass (opaque envelope)
- `h_tr_ms`: Mass → Surface
- `h_tr_is`: Surface → Zone Air
- `h_tr_w`: Windows (directly to zone air)
- `h_ve`: Ventilation (directly to zone air)

### Current CTF Integration (WRONG)

**File**: `src/sim/engine.rs` lines 3618-3649

```rust
// === Add CTF envelope conduction heat flux (if enabled) ===
// SESSION 48: CTF flux goes to MASS energy balance (not zone air)
// This matches the 5R1C model where h_tr_em connects exterior to mass
if let Some(ctf_fluxes) = &ctf_flux_w {
    let slice = phi_m.as_mut();  // ← PROBLEM: Adding to MASS balance
    for (i, &q_flux) in ctf_fluxes.iter().enumerate() {
        if i < slice.len() {
            let q_ctf = q_flux * area;

            // Calculate standard 5R1C exterior conduction for comparison
            let q_5r1c = h_tr_em_i * (t_sol_air_i - t_mass);

            // Add NET CTF flux (CTF - 5R1C) to mass balance
            let q_net = q_ctf - q_5r1c;
            slice[i] += q_net;  // ← PROBLEM: Modifying mass balance
        }
    }
}
```

### Why This Is Wrong

**Issue 1: Wrong Energy Balance**
- CTF flux is added to `phi_m` (mass energy balance)
- But `h_tr_em` still appears in `h_ext` used for zone air calculation
- This creates a **topology mismatch**

**Issue 2: Double Counting with Wrong Sign**
- Mass balance: `phi_m += (q_ctf - q_5r1c)` = `phi_m - 3270 + 278` = `phi_m - 2992`
- Zone air calculation still uses full `h_ext` (including `h_tr_em`)
- Result: Heat transfer path is broken

**Issue 3: Sign Convention Mismatch**
- 5R1C: `h_tr_em * (T_ext - T_mass)` → contributes to zone air temp via network
- CTF: Direct flux to mass → bypasses network topology
- Net effect: Large positive value added (+2991 W) is wrong sign

### The Correct 5R1C Energy Balance

**Zone Air Energy Balance** (derived from network):
```
Q_hvac + phi_ia + phi_st + h_ext*T_ext + h_tr_ms*T_mass
-------------------------------------------------------- = T_zone
                sensitivity
```

Where:
- `h_ext = h_opaque + h_tr_w + h_ve`
- `h_opaque = (h_tr_ms * h_tr_em) / (h_tr_ms + h_tr_em)` (includes h_tr_em!)
- `sensitivity` is network-derived conductance

**Mass Energy Balance**:
```
C_m * dT_mass/dt = phi_m + h_tr_ms*T_zone + h_tr_em*T_ext
```

### What the CTF Integration Should Do

**Option A: Replace h_tr_em in Network**
- CTF calculates `Q_ctf` (exterior → mass)
- This should **replace** `h_tr_em` in the network derivation
- Zone air balance should use modified `h_ext` without `h_tr_em`
- Add `Q_ctf` directly to mass balance (not net difference)

**Option B: Direct Zone Air Injection**
- CTF calculates `Q_ctf` (exterior → zone air, including mass effects)
- Add directly to zone air energy balance (`phi_ia`)
- Remove `h_tr_em` from `h_ext` calculation
- Simpler but less accurate for thermal mass effects

## Evidence from Code

### h_ext Includes h_tr_em

**File**: `src/sim/engine.rs` lines 2328-2333

```rust
// h_opaque includes h_tr_em (exterior → mass conductance)
let h_opaque = (h_tr_is_ms_series.clone() * self.h_tr_em.clone())
    / (h_tr_is_ms_series + self.h_tr_em.clone());

// h_ext = h_opaque + h_tr_w + h_ve
self.derived_h_ext = h_opaque + self.h_tr_w.clone() + self.h_ve.clone();
```

### h_ext Used in Zone Air Calculation

**File**: `src/sim/engine.rs` line 3695

```rust
// Zone air energy balance uses h_ext (which includes h_tr_em)
for (n, h) in num_rest_with_iz.as_mut().iter_mut().zip(h_ext.as_ref().iter()) {
    *n += h * outdoor_temp;  // ← h_tr_em is in h_ext!
}
```

### CTF Flux Added to Wrong Balance

**File**: `src/sim/engine.rs` lines 3618-3649

```rust
// CTF flux added to mass balance (WRONG)
if let Some(ctf_fluxes) = &ctf_flux_w {
    let slice = phi_m.as_mut();  // ← Should be zone air balance
    // ...
    slice[i] += q_net;  // ← Creates topology mismatch
}
```

## Impact on Results

### Peak Loads Degraded Significantly

| Metric | 5R1C Result | CTF Result | Reference | 5R1C Error | CTF Error |
|--------|-------------|------------|-----------|------------|-----------|
| Peak Heating | 1.26 kW | 3.23 kW | 1.80-2.40 kW | -30% | +35% ❌ |
| Peak Cooling | 2.35 kW | 2.89 kW | 1.60-2.10 kW | +12% | +38% ❌ |

**Root cause**: Integration bug creates artificial heat input (+2991 W)

### Annual Energies Still Pass (But Misleadingly)

| Metric | 5R1C Result | CTF Result | Reference |
|--------|-------------|------------|-----------|
| Annual Heating | 1.71 MWh | 1.73 MWh | 1.17-2.04 MWh ✅ |
| Annual Cooling | 2.28 MWh | 2.53 MWh | 2.13-3.67 MWh ✅ |

**Why still passing**: Energy conservation violations average out over 8760 hours

## The Fix: Two Options

### Option A: Proper Network Integration (Recommended)

1. **Remove `h_tr_em` from `h_ext`** when CTF is enabled
2. **Add CTF flux to zone air balance** (not mass balance)
3. **Preserve thermal mass effects** through proper coupling

```rust
// When CTF is enabled, calculate h_ext without h_tr_em
let h_ext_ctf = if self.ctf_enabled {
    // h_ext_ctf = h_tr_w + h_ve (exclude h_opaque which contains h_tr_em)
    &self.h_tr_w + &self.h_ve
} else {
    &self.derived_h_ext  // Standard 5R1C
};

// Add CTF flux to zone air balance (phi_ia), not mass balance
if let Some(ctf_fluxes) = &ctf_flux_w {
    let slice = phi_ia.as_mut();  // ← Zone air balance, not mass
    for (i, &q_flux) in ctf_fluxes.iter().enumerate() {
        if i < slice.len() {
            let area = self.zone_area.as_ref().get(i).copied().unwrap_or(1.0);
            slice[i] += q_flux * area;  // ← Direct addition, not net
        }
    }
}
```

**Pros**: Preserves network topology, thermal mass effects handled correctly
**Cons**: More complex, requires conditional `h_ext` calculation

### Option B: Simplified Direct Injection (Quick Fix)

1. **Keep current approach** but fix sign and balance
2. **Remove h_tr_em from h_ext** unconditionally
3. **Treat CTF as total envelope load**

```rust
// Always remove h_tr_em from h_ext when CTF is enabled
let h_ext_modified = if self.ctf_enabled {
    // Recalculate h_ext without h_tr_em contribution
    let h_opaque_no_ctf = // Series combination without h_tr_em
    h_opaque_no_ctf + &self.h_tr_w + &self.h_ve
} else {
    self.derived_h_ext.clone()
};

// Add CTF flux directly (no net difference)
if let Some(ctf_fluxes) = &ctf_flux_w {
    let slice = phi_ia.as_mut();  // ← Zone air balance
    for (i, &q_flux) in ctf_fluxes.iter().enumerate() {
        if i < slice.len() {
            let area = self.zone_area.as_ref().get(i).copied().unwrap_or(1.0);
            slice[i] += q_flux * area;  // ← Direct, no 5R1C comparison
        }
    }
}
```

**Pros**: Simpler implementation, easier to debug
**Cons**: Loses some thermal mass physics, may need re-tuning

## Verification Steps

After implementing the fix:

1. **Check flux magnitudes**: CTF and 5R1C should be similar order of magnitude
2. **Verify sign convention**: Negative flux = heat leaving zone
3. **Test energy conservation**: `Q_in = Q_out + dE_storage` (< 1% imbalance)
4. **Run Case 900 validation**: Compare with reference range
5. **Profile performance**: Ensure < 3x slowdown acceptable

## Next Steps

1. **Implement Option A** (proper network integration)
2. **Add debug output** for flux comparison at each timestep
3. **Run single-day test** to verify energy conservation
4. **Full year validation** once single-day passes
5. **Document lessons learned** for FD solver integration

## Conclusion

The CTF flux magnitude mismatch is **not a CTF solver bug** but an **integration topology error**. The current implementation adds CTF flux to the mass balance while still using `h_tr_em` in the zone air energy balance, creating a network mismatch.

**Status**: 🔴 BLOCKS Session 49 - Must fix before enabling CTF for all 900-series cases

**Recommendation**: Implement Option A (proper network integration) with careful energy conservation validation before full-year simulation.

---

**Analysis Completed**: 2026-03-27
**Session**: 48 (CTF Solver Audit - Integration Bug Discovery)
**Next**: Implement fix and re-run Case 900 validation
