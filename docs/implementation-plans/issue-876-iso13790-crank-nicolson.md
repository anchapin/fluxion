# Issue #876: ISO 13790 Crank-Nicolson Mass Update — Implementation Plan

## Status: Design Complete, Ready for Implementation

## Summary

Replace the backward Euler mass temperature update in `step_physics_9r4c()` with ISO 13790 §C.4-C.5 Crank-Nicolson formulation. This fixes annual heating overshoot (4.59 → target 1.17-2.04 MWh) by giving the mass a physically correct time constant (~500h instead of ~17h).

## Key Discovery: Gain Routing Already Correct

**ASHRAE 140 sets `solar_distribution_to_air = 0.0`** (thermal_model_core.rs:1553). This means:
- `phi_ia = load_w × conv_frac` (NO solar) — already matches ISO 13790 Eq C.1
- All solar goes to `phi_m` via `beam_to_mass_fraction = 1.0`
- No gain rerouting needed! The previous 78°C failure was NOT caused by gain routing.

## Root Cause of Previous Failure

The 78°C free-float result when trying Crank-Nicolson was likely caused by:
1. Incorrect `phi_m_tot` assembly (wrong conductance values)
2. Using `h_tr_ms` (1300 W/K) instead of `H_tr_3` (~40 W/K) in the coupling term
3. Surface temperature intermediate step incompatible with CN formulation

## Changes Required

### Change 1: Add H_tr_3 to solver cache

**File**: `src/sim/thermal_model_solvers.rs` (update_optimization_cache)
**File**: `src/sim/thermal_model_data.rs` (add field)

Add three new cached fields to `ThermalModelData`:
```rust
pub derived_h_tr_1: T,  // 1/(1/h_ve + 1/h_tr_is) — combined ventilation + surface-air
pub derived_h_tr_2: T,  // h_tr_1 + h_tr_w — adds window conductance
pub derived_h_tr_3: T,  // 1/(1/h_tr_2 + 1/h_tr_ms) — combined air-to-mass (~40 W/K)
```

Compute in `update_optimization_cache()`:
```rust
self.0.derived_h_tr_1 = (h_ve.clone() * h_tr_is.clone()) / (h_ve.clone() + h_tr_is.clone());
self.0.derived_h_tr_2 = self.0.derived_h_tr_1.clone() + self.0.h_tr_w.clone();
self.0.derived_h_tr_3 = (self.0.derived_h_tr_2.clone() * self.0.h_tr_ms.clone())
    / (self.0.derived_h_tr_2.clone() + self.0.h_tr_ms.clone());
```

### Change 2: Add compute_phi_m_tot() helper

**File**: `src/sim/thermal_integration.rs`

```rust
/// ISO 13790 §C.5: Total heat flow to mass node via thermal network.
/// Replaces backward Euler's h_tr_ms × t_s + phi_m.
#[allow(clippy::too_many_arguments)]
pub fn compute_phi_m_tot(
    phi_m: f64,
    phi_st: f64,
    phi_ia: f64,     // includes Q_hc for HVAC zones
    h_tr_em: f64,
    h_tr_w: f64,
    h_tr_1: f64,
    h_tr_2: f64,
    h_tr_3: f64,
    h_ve: f64,
    t_ext: f64,
    t_supply: f64,   // = t_ext for natural ventilation
) -> f64 {
    let inner = phi_st + h_tr_w * t_ext + h_tr_1 * (phi_ia / h_ve.max(1e-10) + t_supply);
    phi_m + h_tr_em * t_ext + h_tr_3 * inner / h_tr_2.max(1e-10)
}
```

### Change 3: Replace backward Euler with Crank-Nicolson in step_physics_9r4c

**File**: `src/sim/thermal_model_physics.rs` (~L2629-2684)

**HVAC path** (zones with heating/cooling):
```
1. Compute t_i_free from 5R1C steady-state (UNCHANGED)
2. Compute Q_hc = h_coeff × (T_set - T_free) (UNCHANGED)
3. phi_ia_total = phi_ia + Q_hc  (HVAC enters air node)
4. Compute phi_m_tot = compute_phi_m_tot(phi_m, phi_st, phi_ia_total, ...)
5. T_m_new = crank_nicolson_iso13790(T_m_old, dt, Cm, H_tr_3, h_tr_em, phi_m_tot)
```

**Free-float path** (zones without HVAC):
```
1. Compute t_i_free from 5R1C steady-state (UNCHANGED)
2. Set t_zone = t_i_free (UNCHANGED)
3. Compute phi_m_tot = compute_phi_m_tot(phi_m, phi_st, phi_ia, ...)
   where phi_ia has NO Q_hc (free-float)
4. T_m_new = crank_nicolson_iso13790(T_m_old, dt, Cm, H_tr_3, h_tr_em, phi_m_tot)
```

### Change 4: Update crank_nicolson_iso13790 signature

**File**: `src/sim/thermal_integration.rs` (L138-176)

Current signature uses `phi_m_tot` directly. Keep this — the function is already correct.
Just need to ensure the fallback for negative denominator is robust.

## Order of Operations (Critical)

```
FOR EACH TIMESTEP:
  1. Compute gains: phi_ia, phi_st, phi_m (UNCHANGED, already ISO-compliant)
  2. Compute t_i_free_5r1c using OLD mass temperature (UNCHANGED)
  3. For free-float: set t_zone = t_i_free, Q_hc = 0
  4. For HVAC: compute Q_hc from coefficient formula (UNCHANGED)
  5. Compute phi_ia_total = phi_ia + Q_hc (HVAC zones) or phi_ia (free-float)
  6. Compute phi_m_tot using phi_ia_total + network conductances
  7. Update mass: T_m_new = crank_nicolson_iso13790(T_m_old, dt, Cm, H_tr_3, h_tr_em, phi_m_tot)
  8. Store new mass temperature
```

No chicken-and-egg: t_i_free uses OLD mass temp, Q_hc depends on t_i_free (OLD mass), mass update uses Q_hc.

## Data Mapping

| ISO 13790 | Our Field | Available? |
|-----------|-----------|------------|
| H_tr_em | self.0.h_tr_em | Yes |
| H_tr_w | self.0.h_tr_w | Yes |
| H_ve_adj | self.0.h_ve | Yes |
| H_tr_is | self.0.h_tr_is | Yes |
| H_tr_ms | self.0.h_tr_ms | Yes |
| H_tr_1 | NEW: derived_h_tr_1 | To add |
| H_tr_2 | NEW: derived_h_tr_2 | To add |
| H_tr_3 | NEW: derived_h_tr_3 | To add |
| T_ext | outdoor_temp | Yes |
| T_supply | outdoor_temp (= T_ext) | Yes |
| phi_m | phi_m.as_ref()[i] | Yes |
| phi_st | phi_st.as_ref()[i] | Yes |
| phi_ia | phi_ia.as_ref()[i] | Yes |
| Q_hc | hvac_for_temp_calc.as_ref()[i] | Yes |
| C_m | thermal_capacitance.as_ref()[i] | Yes |

## Expected Results

| Metric | Current (BE) | Expected (CN) | Reference |
|--------|-------------|---------------|-----------|
| Annual heating | 4.59 MWh | ~1.5-2.0 MWh | 1.17-2.04 |
| Annual cooling | ~2.1 MWh | ~2.5-3.5 MWh | 2.13-3.67 |
| 900FF max | 44.64°C | ~44-46°C | 41.8-46.4 |
| 900FF min | -0.57°C | ~-5 to -12°C | -12.0 to -1.60 |
| H/C ratio | 2.18 | ~0.5 | reference ~0.52 |
| Mass τ | ~17h | ~500h | ISO 13790 |

## Risk Mitigation

1. **Incremental testing**: After each change, run `cargo test --test ashrae_140_case_900`
2. **Free-float preservation**: 900FF must stay in 41.8-46.4°C range. If it breaks, check phi_m_tot assembly
3. **Stability**: C-N can produce negative denominators if `0.5×(H_tr_3 + h_tr_em) > Cm/dt`. Already handled by existing fallback
4. **Gain routing**: Already correct for ASHRAE 140 (solar_distribution_to_air = 0.0). DO NOT modify gain routing

## Implementation Order

1. Add H_tr_1/2/3 fields to data + solver cache → compile check
2. Add compute_phi_m_tot() to thermal_integration.rs → unit test
3. Replace backward Euler call with Crank-Nicolson + phi_m_tot in physics.rs → run Case 900
4. Iterate on any failures
5. Run full lib tests (2457 must pass)
6. Commit

## Estimated Effort

4-6 hours (medium complexity, well-researched)
