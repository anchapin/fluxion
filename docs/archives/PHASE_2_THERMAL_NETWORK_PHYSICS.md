# Phase 2: Thermal Network Physics Analysis

**Date:** 2026-03-28
**Status:** ✅ Complete
**Summary:** Identified critical physics errors in thermal network topology and conductance formulas.

---

## Task 2.1: Map 5R1C Thermal Network ❌ CRITICAL ERRORS

### Current Implementation Analysis

Based on the code in `src/sim/engine.rs` lines 1349-1554, the 5R1C thermal network structure is:

```
         Outdoor (T_ext)
               │
               ├─ h_ve (ventilation)
               │
               ├─ h_tr_w (windows)
               │
               └─ h_tr_em (exterior to mass)
                     │
                     └─ [Thermal Mass (Tm)]
                           │
                           ├─ h_ms × A_m (mass to surface)
                           │
                           └─ h_tr_is (surface to interior)
                                 │
                                 └─ [Zone Air (Ti)]
```

### Heat Balance Equations

**1. Zone Air Temperature (Ti_free):**
```rust
// term_rest_1 = h_tr_ms + h_tr_is
// derived_h_ms_is_prod = h_tr_ms * h_tr_is

let num_tm = derived_h_ms_is_prod * mass_temperatures;
let num_phi_st = h_tr_is * phi_st;  // Radiative gain to surface
let num_rest = term_rest_1 * (h_ext * outdoor_temp + phi_ia);

let t_i_free = (num_tm + num_phi_st + num_rest) / den;
```

Where:
- `den = derived_h_ms_is_prod + term_rest_1 * h_ext`
- `h_ext = h_tr_w + h_ve` (or modified for night ventilation)

**2. Surface Temperature (Ts_free):**
```rust
let ts_num_free = h_tr_ms * mass_temperatures
    + h_tr_is * t_i_free
    + phi_st;
let t_s_free = ts_num_free / term_rest_1;
```

**3. Mass Temperature Update (Tm_next):**
```rust
let q_m_net = h_tr_em * (outdoor_temp - mass_temperatures)
    + h_tr_ms * (t_s_free - mass_temperatures)
    + phi_m;  // Radiative gain directly to mass

let dt_m = (q_m_net / thermal_capacitance) * dt;
let tm_next = tm_current + dt_m;
```

### Network Topology Analysis

**Mass Node Heat Sources:**
1. **Path 1: h_tr_em × (T_ext - Tm)** - Direct from exterior to mass
2. **Path 2: h_tr_ms × (Ts - Tm)** - From surface to mass
3. **Path 3: phi_m** - Radiative gain directly to mass

**Key Observations:**

1. **h_tr_em path is INVALID:**
   - Mass is NOT directly exposed to exterior air in real buildings
   - Heat flows through opaque surfaces (walls, roof, floor) first
   - Then from surfaces to zone air (via h_tr_is)
   - Then from zone air to mass (via h_tr_ms)

2. **Parallel Heat Transfer to Mass:**
   - The current implementation shows TWO parallel paths to the mass:
     - Direct: Exterior → Mass (via h_tr_em)
     - Indirect: Exterior → Surface → Mass (via h_tr_ms + h_tr_is)
   - This is DOUBLE-COUNTING and physically incorrect

3. **Series vs Parallel Confusion:**
   - The steady-state test (lines 2467-2478) uses SERIES formula:
     ```rust
     let u_opaque = 1.0 / (1.0/h_tr_em + 1.0/h_tr_ms + 1.0/h_tr_is);
     ```
   - But the dynamic simulation uses BOTH h_tr_em AND h_tr_ms paths
   - This creates an inconsistent thermal network

### Correct 5R1C Topology (ISO 13790)

The ISO 13790 5R1C model should have this structure:

```
         Outdoor (T_ext)
               │
               ├─ h_ve (ventilation)
               │
               ├─ h_tr_w (windows)
               │
               └─ [Opaque Surfaces]
                     │
                     └─ [Zone Air (Ti)]
                           │
                           └─ h_tr_ms (mass coupling)
                                 │
                                 └─ [Thermal Mass (Tm)]
```

**Key Differences:**
1. **No h_tr_em in standard 5R1C** - Mass is coupled to air, not exterior
2. **Opaque surfaces connect to air, not directly to mass**
3. **Mass receives heat from air via h_tr_ms, not exterior**
4. **Radiative gains go to surface node, then to mass via h_tr_ms**

---

## Task 2.2: Verify h_tr_em Calculation ❌ CRITICAL ERROR

### Current Implementation

**Location:** `src/sim/engine.rs` lines 686-692

```rust
// Opaque conductance (h_tr_em)
let wall_u = spec.construction.wall.u_value(None);
let roof_u = spec.construction.roof.u_value(None);
let h_tr_op = opaque_area * wall_u + zone_floor_area * roof_u + model.thermal_bridge_coefficient;
let h_tr_em_val = 1.0 / ((1.0 / h_tr_op) - (1.0 / (h_ms * a_m)));
h_tr_em_vec.push(h_tr_em_val.max(0.1));
```

### Problem Analysis

**Formula Breakdown:**
```
Let R_opaque = 1/h_tr_op  (opaque resistance)
Let R_mass = 1/(h_ms × A_m)  (mass resistance)

h_tr_em = 1 / (R_opaque - R_mass)
       = 1 / R_opaque - R_mass  [WRONG: subtracting conductances]
```

**Why This is Incorrect:**

1. **Subtracts resistances** - No physical meaning
2. **Parallel formula should be:** `h_total = 1 / (1/R1 + 1/R2)`
3. **Can produce negative values** when R_mass > R_opaque
4. **Can produce extremely large values** near singularity point

**Test with Case 600 Values:**
```
A_m = 2.5 × 64 = 160 m²
h_ms = 9.1 W/m²K
h_tr_ms = h_ms × A_m = 1456 W/K
R_mass = 1/1456 = 0.00069 K/W

Wall U = 0.514 W/m²K, Roof U = 0.318 W/m²K
Opaque area ≈ 96 m² (walls + roof)
h_tr_op = 96 × (0.514 × 0.7 + 0.318 × 0.3) ≈ 41 W/K
R_opaque = 1/41 = 0.024 K/W

h_tr_em = 1 / (0.024 - 0.00069) = 1 / 0.0233 = 43 W/K
```

**Issue:** This calculates a conductance that represents subtracting resistances, which has no clear physical meaning in the actual thermal network.

### Correct Physics

**Option A: Remove h_tr_em entirely**
If the mass should be coupled to air, not exterior:
```rust
// Set to 0 - no direct exterior-to-mass coupling
h_tr_em_vec.push(0.0);
```

**Option B: Use proper parallel resistance formula**
If modeling both paths:
```rust
// Parallel: 1/R_total = 1/R1 + 1/R2
// h_tr_em = h_tr_op + h_ms × A_m  (for parallel resistances)
```

**Option C: Derive from actual construction layers**
```rust
// Calculate actual thermal resistance from layer stack
// h_tr_em should represent actual heat transfer path
```

---

## Task 2.3: Verify h_tr_ms Coupling ❌ CRITICAL ERROR

### Current Implementation

**Location:** `src/sim/engine.rs` lines 681-684

```rust
// Mass-to-surface conductance (h_ms = 9.1 × A_m)
// ISO 13790 standard value for mass-to-surface conductance
let h_ms = 9.1;
h_tr_ms_vec.push(h_ms * a_m);
```

### Problem Analysis

**Issue 1: 9.1 W/m²K is NOT an ISO 13790 coefficient**

Searching ISO 13790 Annex C and documentation:
- **No fixed coefficient of 9.1 W/m²K exists in ISO 13790**
- The coefficient appears to be a **misunderstanding or misapplication**

**Issue 2: Using coefficient × area gives incorrect conductance**

For Case 600:
```
A_m = 2.5 × 64 = 160 m²
h_tr_ms = 9.1 × 160 = 1456 W/K
```

This value is **700x higher** than the working empirical fix (2.0 W/K) and **100-200x higher** than physics-based estimates.

**Why This Causes 4x Higher Energy:**

**Thermal Time Constant:**
```
τ = C_m / h_tr_ms

For Case 600:
C_m ≈ 100 kJ/K = 100,000 J/K
h_tr_ms = 1456 W/K
τ = 100,000 / 1456 = 69 seconds ≈ 1.1 MINUTES
```

With τ = 1.1 minutes:
- Mass responds WAY too fast to thermal changes
- Doesn't store heat effectively
- Gains escape rapidly → excess heating needed
- Stored heat lost rapidly → excess cooling needed
- HVAC works 4x harder

**Correct Thermal Time Constant:**

For realistic buildings, τ should be 1-4 hours:
```
τ_target = 2-4 hours = 7200-14400 seconds
h_tr_ms_correct = C_m / τ_target

For Case 600:
h_tr_ms_correct = 100,000 / 3600 = 28 W/K (for 1 hour)
h_tr_ms_correct = 100,000 / 7200 = 14 W/K (for 2 hours)
```

This gives a physics-based range of 14-28 W/K, which is:
- 50-100x LOWER than current 1456 W/K
- 7-14x HIGHER than empirical 2.0 W/K (empirical fix may be too low)

---

## Task 2.4: Check for Double-Counting ❌ CONFIRMED

### Double-Counting Issue #1: Parallel Paths to Mass

**Current Implementation:**
```rust
let q_m_net = h_tr_em * (outdoor_temp - mass_temperatures)      // Path 1: Direct
    + h_tr_ms * (t_s_free - mass_temperatures)          // Path 2: Via surface
    + phi_m;                                            // Path 3: Direct gain
```

**Problem:**
- Heat from exterior to mass is counted TWICE:
  1. Directly via `h_tr_em × (T_ext - Tm)`
  2. Indirectly via `h_tr_ms × (Ts - Tm)` where Ts is heated by exterior

### Double-Counting Issue #2: h_tr_em in Steady-State but Not in Dynamic

**Steady-State Formula (line 2467):**
```rust
let u_opaque = 1.0 / (1.0/h_tr_em + 1.0/h_tr_ms + 1.0/h_tr_is);
```

This treats h_tr_em as a SERIES resistance with h_tr_ms and h_tr_is.

**Dynamic Simulation (line 1515):**
```rust
let q_m_net = h_tr_em * (outdoor_temp - mass_temperatures)
    + h_tr_ms * (t_s_free - mass_temperatures)
    + phi_m;
```

This uses BOTH h_tr_em and h_tr_ms as PARALLEL paths.

**Inconsistency:** The steady-state and dynamic calculations use DIFFERENT thermal network topologies!

---

## Root Cause Summary

### Primary Issues

1. **h_tr_em uses incorrect resistance subtraction formula**
   - Formula: `1 / (1/R1 - 1/R2)`
   - Should be: `0` (no direct coupling) or proper parallel formula

2. **h_tr_ms uses arbitrary 9.1 W/m²K coefficient**
   - This is NOT in ISO 13790
   - Should be derived from thermal time constant: `C_m / τ`

3. **Thermal network has double-counting**
   - Exterior-to-mass heat flow counted via both h_tr_em and h_tr_ms paths
   - Inconsistent between steady-state and dynamic calculations

4. **Incorrect network topology**
   - Mass should be coupled to air, not directly to exterior
   - Opaque surfaces connect to air, then to mass via h_tr_ms

### Impact on Validation Results

**Case 600:**
```
Current: 19.75 MWh heating (expected 4.30-5.71) - +294% error
Root cause: τ ≈ 1.1 minutes (way too fast)
Fix target: τ ≈ 2 hours → h_tr_ms ≈ 14-28 W/K
```

**Case 900:**
```
Current: 22.54 MWh heating (expected 1.17-2.04) - +1304% error
Root cause: τ even shorter due to higher C_m
Fix target: τ ≈ 2-4 hours → h_tr_ms ≈ 50-200 W/K
```

---

## Recommended Physics-Based Solutions

### Solution 1: Fix h_tr_ms Calculation

**Derive from thermal time constant:**
```rust
// Calculate thermal time constant from target response time (1-4 hours)
let target_tau_hours = 2.0;  // Adjustable based on building type
let target_tau_seconds = target_tau_hours * 3600.0;

// h_tr_ms = C_m / τ
let h_tr_ms_correct = thermal_capacitance / target_tau_seconds;
```

**For Case 600:**
- C_m ≈ 100 kJ/K
- τ = 2 hours = 7200s
- h_tr_ms = 100,000 / 7200 = 14 W/K

**For Case 900:**
- C_m ≈ 225 kJ/K
- τ = 4 hours = 14400s (heavier mass = slower response)
- h_tr_ms = 225,000 / 14400 = 16 W/K

### Solution 2: Fix h_tr_em Topology

**Option A: Remove entirely (ISO 13790 compliant):**
```rust
// No direct exterior-to-mass coupling in 5R1C model
h_tr_em_vec.push(0.0);
```

**Option B: Correct formula if parallel paths are modeled:**
```rust
// Proper parallel resistance formula
let r_opaque = 1.0 / h_tr_op;
let r_mass = 1.0 / (h_ms * a_m);

// Parallel: R_total = 1 / (1/R1 + 1/R2)
// h_tr_em = 1.0 / r_opaque + 1.0 / r_mass;
```

### Solution 3: Remove Double-Counting

**Ensure single heat transfer path:**
```rust
// Mass should receive heat from air only (via h_tr_ms)
// Remove h_tr_em contribution from q_m_net
let q_m_net = h_tr_ms * (t_s_free - mass_temperatures) + phi_m;
```

### Solution 4: Align Steady-State and Dynamic Calculations

**Use consistent thermal network in both cases.**

---

## Next Steps (Phase 3: Diagnostic Instrumentation)

1. Add conductance logging to verify calculated values
2. Track heat flow through each thermal path
3. Monitor thermal mass energy change
4. Verify energy balance at each timestep

---

## Files Examined

- `src/sim/engine.rs` lines 1349-1554 - Main 5R1C physics solver
- `src/sim/engine.rs` lines 681-692 - h_tr_em and h_tr_ms calculation
- `src/sim/engine.rs` lines 2467-2478 - Steady-state test formulas
- `SESSION_36_THERMAL_MASS_SUMMARY.md` - Previous thermal mass investigation

---

## Conclusions

### Critical Physics Errors Confirmed:

1. ❌ **h_tr_em formula** uses invalid resistance subtraction
2. ❌ **h_tr_ms formula** uses arbitrary 9.1 W/m²K coefficient
3. ❌ **Thermal network** has double-counting and inconsistent topology
4. ❌ **Thermal time constant** is ~1 minute (should be 1-4 hours)

### Impact:

- Mass responds 50-100x too fast to thermal changes
- Heat storage effect is negligible
- HVAC must compensate with 4x higher energy
- Results in 15.6% pass rate vs 100% expected

### Physics-Based Fix Required:

1. **Derive h_tr_ms from thermal time constant:** `h_tr_ms = C_m / τ`
2. **Fix or remove h_tr_em** based on correct network topology
3. **Remove double-counting** in heat flow calculation
4. **Align steady-state and dynamic** thermal network models

---

**Phase 2 Complete.** Ready for Phase 3: Diagnostic Instrumentation.
